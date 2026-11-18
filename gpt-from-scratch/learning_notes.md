# GPT from scratch: learning notes

What each step taught while building Karpathy's GPT by hand. The steps, settings and losses are in [plan.md](plan.md). These notes cover the why: the questions asked along the way, and the predictions that went wrong, since those taught the most.

Status: step 5c done 2026-11-18 (started 2026-11-17). A section is added after each step, before its commit.

## Setup

- **The global Python has `torch 2.15.0+cpu`, the CPU-only build.** It trains without complaint, just on the CPU and hours slower. Check `torch.cuda.is_available()` in the environment you actually run. This folder's venv has `torch 2.15.0+cu132`.
- **CUDA versions:** the driver supports CUDA 13.3, but PyTorch publishes builds only up to 13.2. A driver runs builds made for older CUDA versions, so cu132 works.
- **Hardware:** Karpathy trained the final model for about 15 minutes on an A100 (a data-centre GPU, 40 or 80 GB). The RTX 4070 SUPER's 12 GB is enough for his full settings, just slower.

## Choosing the training text

What makes a text good for training:

| | Why |
|---|---|
| About 1 MB or more | Much less and a 10.8M-parameter model memorises it: validation loss rises while training loss keeps falling |
| One style throughout | The last 10% becomes the validation set. An index, a licence or a bibliography at the end makes the validation loss look bad for reasons unrelated to the model |
| A modest character set | Every distinct character gets its own row in the table. Hundreds of rare symbols waste rows |

- **The format:** EPUB beats PDF. An EPUB is a zip of HTML pages, so paragraphs are whole `<p>` elements. A PDF mixes page numbers and running headers into the text and splits words at line breaks. A single Gutenberg HTML file beats both: one file, and the boilerplate is marked (`section.pg-boilerplate`).
- **Copyright:** this repo is public. Training on a text locally is fine, but pushing it would republish it, so `input.txt` and downloaded books are in `.gitignore`.
- **Vocabulary prediction (open):** my first guess was that a book's character set would be the same size as Shakespeare's 65, then 80. Shakespeare's 65 are hand-typed ASCII: newline, space, `!$&',-.3:;?` and the 52 letters, with no digits except `3`. A typeset book adds curly quotes, dashes, digits and accented foreign names. To check when Freud's text is extracted.

## Step 1: data and tokenizer

- **A character-level tokenizer** is two dicts: character → id (`stoi`) and id → character (`itos`). The vocabulary is the sorted unique characters, 65 for Shakespeare.
- **The split is by position:** the first 90% trains and the last 10% validates. So the validation text is a continuous part of the book the model never saw.

## Step 2: `get_batch`

- **`torch.randint`'s upper bound is exclusive**, like Python's `range`. Passing `len(data) - block_size` keeps `start + block_size + 1` inside the data, which `y` needs.
- **`y` is `x` shifted by one character.** Each of the 8 positions is its own training example: predict character t+1 from characters up to t.
- **`idx` is B blocks of T characters** (B × T), not "one batch of 8 blocks." In training it's 32 × 8. In `generate` it's 1 × 1, then 1 × 2, 1 × 3…

## Step 3: the bigram model

**What `nn.Embedding(65, 32)` does internally:**
1. Creates one tensor, `weight`, of shape (65, 32): one row per character.
2. Fills it from a normal distribution (mean 0, std 1).
3. Wraps it in `nn.Parameter`, so `model.parameters()` returns it, AdamW updates it, and `.to(DEVICE)` moves it.

Its `forward` is plain indexing, `weight[idx]`, like `table[id]` on a Java `float[65][32]`. It's the same as multiplying a one-hot vector by the matrix, without the multiplications by zero.

- **A row's meaning comes from the loss, not the layer.** In the bigram, row `i` holds 65 scores, read as "after character `i`, how likely is each next character," only because cross-entropy compares that row with the actual next character.
- **Only the rows in the batch get a gradient.** A character missing from the batch gives its row a gradient of zero. A used row changes in all 65 columns, because softmax ties the scores together. With AdamW the unused rows still drift a little: weight decay shrinks every weight each step, and momentum keeps moving rows that were used earlier.
- **The initial loss:** a uniform guess gives −ln(1/65) ≈ 4.17. Mine was 5.04 and Karpathy's 4.87, because the random N(0, 1) rows make confident random guesses, which cost more than uniform ones.
- **`F.cross_entropy` takes raw logits** and applies log-softmax itself, so you don't apply softmax first.
- **`model(x)` calls `forward` for you:** `nn.Module.__call__` does PyTorch's bookkeeping (hooks), then calls `forward`, like a Spring proxy around a bean. So you call `model(...)`, never `forward` directly. `targets` is optional because `generate` has none.

## Step 4: training

- **`estimate_loss` vs. the loss from `forward`:** it is the same loss, averaged.

| | Loss in the training loop | `estimate_loss()` |
|---|---|---|
| Behind it | one batch | 200 batches per split |
| Splits | train only | train and val |
| Gradients | needed for `backward()` | not needed: `@torch.no_grad()` |
| Use | one update step | an honest progress report and the overfitting check |

  The single-batch loss is "a very noisy measurement" (Karpathy, 39:50).
- **With Adam, the number of steps decides how far training gets, not the characters seen.** Batch 4 for 5,000 steps gave 2.89; batch 4 for 10,000 steps gave 2.54, the floor. Adam divides each gradient by its own running size, so each step moves a weight by about `lr` whatever the batch size. A bigger batch gives a less noisy direction, not a bigger step. (Claude first explained the 2.89 by characters seen, which was wrong; the 10,000-step run showed it.)
- **The bigram floor (about 2.5):** the model sees only the current character. After `th` and after `sh` it sees just `h`, so both get the same guess. Once the table is right, more training doesn't help.
- **Karpathy's learning rate changes twice:** 1e-3 in the notebook bigram, 1e-2 in `bigram.py` (a tiny model tolerates big steps), back to 1e-3 for attention ("the self attention can't tolerate very very high learning rates").
- **Bugs caught in review:** `{iter}` printed Python's built-in `iter` function instead of the loop variable `i`; `EVAL_INTERVAL` was used as the number of eval batches, which ties two unrelated settings together; asserts on `x` and `y` at the end of the file crashed, because those exist only inside `get_batch`.

## Step 5a: `n_embd` and `lm_head`

- **A level of indirection:** a row is now a 32-number description of the character, and `lm_head` turns it into 65 scores. The (B, T, 32) tensor in between is where the rest of the Transformer works: position, attention, feed-forward and blocks all take (B, T, 32) and return it. In Spring terms: an internal object between the database row and the response, with a mapper at the end.
- **Prediction, wrong:** "better, since the linear layer adds weights." It's about the same. Follow one position from `idx` to `logits`: its scores still depend only on the current character, so it's still a bigram table, and a bigram table can't beat the best bigram table. **Weights help only when the model has information they can use.**
- **Counting parameters:** `nn.Linear(32, 65)` has 33 numbers per output score (32 weights plus a bias), times 65 scores, so 2,145. PyTorch stores the weight as (out, in) = (65, 32), plus a separate bias of 65. `nn.Linear` has a bias unless you pass `bias=False`. The total is 2,080 (token table) + 2,145 = 4,225, exactly the bigram's 65 × 65, so the capacity is the same. (I first said 33, then 4,290, which counted the bias twice.)
- **Result:** 2.497, at the bigram's floor (2.49), as predicted. It got there faster than the plain bigram at the same `lr`: `lm_head` is shared by every character, so every step trains it, and its small initial weights start the predictions near uniform.

## Step 5b: position embedding

- **A second table, one row per position** (8 × 32), added to the character's row:

| Character at position | Token row + position row | x |
|---|---|---|
| `t` at 2 | [2, 1, 0] + [0, 0, 0.3] | [2, 1, 0.3] |
| `t` at 0 | [2, 1, 0] + [0.1, 0, 0] | [2.1, 1, 0] |

  (3 numbers instead of 32, made up.) The same character in different places gives different vectors.
- **The lookup is the same as for tokens:** `torch.arange(T)` makes the ids 0 to T−1 (stopping before T, like `range`), and each id picks a row of the table. The ids are row numbers, not the values inside the rows: each row holds 32 learned numbers of its own, random at the start (N(0, 1)) like the token rows. There's no loop: all T rows are looked up in one call. In training T = 8, so the lookup returns the whole table in order.
- **Each position adds its own pattern, number by number:** number c of the token row plus number c of the position row, for c = 0 to 31. It is not one constant (like 0.1) added to all 32 numbers; that was only a made-up example to make the change easy to see.
- **Broadcasting:** `pos_emb` is (T, 32) with no batch dimension, so PyTorch treats it as (1, T, 32). Row t of the position table is added to the token vector at position t **in every block**, not to block t: the 8 position rows are reused across all 32 blocks. In Java terms, `out[b][t][c] = tokTable[ids[b][t]][c] + posTable[t][c]`.
- **Adding, not concatenating:** adding keeps the vector at 32 numbers, so everything after it stays the same size. Both tables are learned, so training arranges them so that the character and the position can both still be read from the sum.
- **Gradient through the addition:** `x = tok_emb + pos_emb` copies the gradient at each position to both rows, unchanged (see "Gradients through an addition" below).
- **Prediction, wrong:** "yes, position is more information," then "it tells what comes before position 3." Position 3 says that three characters come before, not which ones. And chunks start at random offsets, so the next character after position 3 is as likely to be `e` as after position 5. Position alone predicts nothing: "it's all translation invariant at this stage" (Karpathy, 1:01:37).
- **Position can't learn character facts.** Question asked: if position 0 always added the same numbers, and `b` is usually followed by `a`, would position help learn that? No: "`a` after `b`" is learned by `b`'s token row, which is added to every `b` at any position. Position 0's row is added to whatever character sits at position 0, so it can't carry anything about `b`. Since `lm_head` is linear, `lm_head(tok + pos) = lm_head(tok) + W @ pos`: the position row only adds a fixed set of 65 score shifts at that position, for every character.
- **When position alone would help:** if every chunk started at the same kind of place, e.g. right after a newline, position k would mean "the k-th character of a line," and early positions predict capitals and `:` (speaker names like `ROMEO:`). Taking chunks in order (starting at 0, 8, 16…) is not enough, since multiples of 8 don't line up with anything in the text. Karpathy keeps random starts anyway: `generate` crops to the last 8 characters, which can start anywhere, so training has to cover every starting point.
- **So why add it now:** it's for attention (5c). On its own, attention sees the earlier characters as an unordered set. The position numbers added to each embedding are what let it tell "the character right before me" from "the character 7 back." In the real model, position means "how far back is that other character," not "where am I in the text."
- **T is not always `block_size`.** `B, T = idx.shape` reads T from the input. In training T is always 8, but in `generate` `idx` grows by one character per loop: T = 1, 2, … 501. That's why the lookup uses `torch.arange(T)`, not `torch.arange(BLOCK_SIZE)`.
- **The crash at T = 9:** the table has rows 0 to 7 only, and its shape is fixed in `__init__` (the optimizer holds that exact tensor). So position 8 is out of range, like Java's `ArrayIndexOutOfBoundsException`. The table doesn't grow and the index doesn't wrap (my first two guesses). On the GPU it can show up as a CUDA "device-side assert".
- **The fix:** pass the **last** 8 characters, `idx[:, -BLOCK_SIZE:]`, to the model, and keep appending to the full `idx`. The last 8, because the next character depends on the most recent ones, and the last time step must be the newest character.
- **Result:** 2.503 with 4,481 parameters (+256 for the position table): the same as 5a within noise, as worked out.

## Step 5c: one self-attention head

### What an average loses

Making each position the average of itself and the earlier ones gives it context, but it's "extremely lossy":

| What the average loses | The fix |
|---|---|
| Order and position: `abc` and `cba` average the same | Position embeddings (5b) |
| Every earlier character counts equally | Self-attention: weights computed from the data (query · key) |

### How attention learns

Transpose, `masked_fill` and softmax are fixed plumbing that knows nothing about language. What the head knows is in the three learned layers:

| Line | Learned? | Job |
|---|---|---|
| `k = self.key(x)`, `q = self.query(x)` | yes | "what I contain" and "what I'm looking for" |
| `q @ k.transpose(-2, -1)` | no | every query's dot product with every key; the transpose lines up the shapes |
| `* head_size**-0.5` | no | keeps the scores small, so softmax doesn't jump to one position |
| `masked_fill(tril == 0, -inf)` | no | blocks the future |
| `softmax` | no | weights that sum to 1 |
| `v = self.value(x)` | yes | what a position passes on if it's picked |
| `wei @ v` | no | each position's weighted summary of the relevant past |

Nobody chooses what the channels mean. If "the previous character was a vowel" helps, the gradient pushes the key to put a large number in some channel for vowels and the query to look for that channel; their dot product rises and softmax picks that position. Any pattern that lowers the loss gets strengthened. The structure gives the possibility, the data fills it in, the same way the Digit Recognizer's filters learned to detect strokes.

**CNN vs. attention:** in both you choose how many channels (`Conv2d(1, 32, 3)`, `head_size = 16`), never what they mean. The difference is what's built in. A CNN knows a pixel is brightness and that neighbours belong together. Attention knows almost nothing: a character id is just a number, and which positions get combined is learned per input. That's why attention needs more data, and why it can find patterns no one built in.

### The code

- **`head_size**-0.5`:** a negative exponent means "one over," so it's 1 / √head_size, written as a multiplication (`Math.pow(headSize, -0.5)` in Java). For 32 it's 0.177.
- **`tril` is a buffer** (`register_buffer`), not a parameter: the optimizer never changes it, but `.to(DEVICE)` still moves it. `tril[:T, :T]` crops it for `generate`, where T can be less than 8.
- **Parameters:** 7,553, which is 5b's 4,481 plus 3 × (32 × 32) for key, query and value (no biases).

### The checks

**Rows of `wei` sum to 1, and everything above the diagonal is 0.** `wei` is a local variable in `Head.forward`, so the head stores it (`self.wei = wei`) for the check to read after one batch. Then `wei.sum(dim=-1)` (one sum per row, shape (B, T)) is compared with ones, and `torch.triu(wei, diagonal=1)` (the part above the diagonal) with zeros. The check runs on a fresh batch: `generate` ran last, so `self.wei` still held generation's (1, 8, 8).

| Check | Value | Exact? | Compare with |
|---|---|---|---|
| Above the diagonal | e^(−inf) = 0 | yes: 0 is exact in floats | `== 0` |
| Row sums | 8 rounded divisions added up | no: float32 keeps about 7 digits, so 0.99999994 or 1.0000001 | `torch.allclose` |

The same reason Java compares doubles with a tolerance: `0.1 + 0.2 == 0.3` is False.

**No peeking at the future:** run a batch, change only the last character of each block, run again. Positions 0 to 6 can't see position 7, so their logits must be identical (`torch.equal(logits_1[:, :-1], logits_2[:, :-1])`); position 7 sees its own new character, so its logits must change in every block.

- **The first version tested 1 block of 32.** `x_block = xb[:, -1]` is the last column, shape (32,), so `x_block[7]` was block 7's last character only. Both asserts still passed, because they were loose: 31 blocks had identical inputs, and `not torch.equal(...)` passes when any single value differs.
- **The fix:** change the whole column on a copy (`xb2 = xb.clone()`; `xb[:, -1]` is a view, so writing to it changes `xb` itself), with `torch.where(last != 0, last // 2, 1)`. `x if cond else y` works on one number but raises "Boolean value of Tensor with more than one value is ambiguous" on a tensor; `torch.where` picks element by element. Then check the last position block by block: `(logits_1[:, -1] != logits_2[:, -1]).any(dim=-1).all()`.
- **Testing the test:** with the `masked_fill` line removed, the `wei` check fails, and with that check also removed, the no-peeking check fails. A check is only worth having if it fails when the thing it guards is broken.

### Result

Validation **2.4146** (train 2.3970), against 2.50 for 5b and Karpathy's 2.4. The first attention head beats the bigram's floor.

A trained `wei` (the first block of a batch) puts most of each row's weight on the diagonal, the position itself: 0.91, 0.91, 0.71, 0.99, 0.69, 0.97, with a little on the one or two characters before. Open: why.

### Why one head only takes the loss from 2.5 to 2.4

I picked A and C, ruled out B and E correctly, and missed D and F.

| Limit | Fix |
|---|---|
| A: one head, so one weighted mix of the past | Multi-head (2.28) |
| C: 8 characters of context | `block_size = 256` |
| D: no processing after gathering | Feed-forward (2.24) |
| F: 32 numbers per position | `n_embd = 384` |

- **Why D matters:** `wei @ v` is a weighted average and `lm_head` is linear, so the model can only add evidence up. It can't express "the character before me was `t` **and** I'm `h`, so next is `e`": that needs a nonlinearity. In the Digit Recognizer, adding one ReLU layer took the logistic baseline's 91.31% to 96.60%. Two linear layers without a ReLU collapse into one.
- **Ruled out:** B (the mask hides the future, but the future isn't available when generating anyway) and E (without the scaling, softmax peaks at the highest score).

## Ahead: from the video, discussed before building

To move into the step that builds each one.

### Gradients through an addition

At `z = x + y`, the gradient is **copied** to both inputs, not split: ∂z/∂x = ∂z/∂y = 1, so both get the full g. With x = 3, y = 4 and L = z², both get 14. Multiplication instead scales each by the other input. When a value is used in two places, its gradients from both uses are summed.

### Residual connections (step 7)

- **Prediction:** with `x = x + block(x)` and a block that passes almost no gradient back, the incoming x still gets about **g**: the full gradient through the direct path, plus nearly nothing through the block. (I wasn't sure at first; drawing the two paths answered it.)
- **The problem without them:** a chain of blocks multiplies their local gradients. At 0.5 per block, 6 blocks give 0.016 and 20 give 0.000001: the early layers stop learning.
- **The fix:** the local gradient of `x + f(x)` is 1 + f′, so the product always contains 1 × 1 × … × 1, a direct path to the embeddings. Karpathy's "gradient super highway."
- **Why "residual":** the blocks start small, so `x + f(x) ≈ x` and the network trains like a shallow one first. Each block learns a correction to x, not a new x.
- **The residual stream:** x flows from the embeddings to `lm_head`, and each layer reads from it and adds its result back. It also keeps the character's own embedding in the stream: `x = self.sa_head(x)` in 5c replaces it.
- **What it adds to the code:** `proj` (a `Linear` back to `n_embd`, so the heads' output can be added to x), and pre-norm, `x = x + self.sa(self.ln1(x))`, with the layer norm on the branch so the direct path stays a pure addition.
- From ResNet (2015), which trained 152-layer image networks. Karpathy's model has 6 blocks; GPT-3 has 96.
