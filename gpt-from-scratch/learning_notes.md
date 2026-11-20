# GPT from scratch: learning notes

How the GPT was built, step by step, following Andrej Karpathy's "Let's build GPT": the decision taken at each step and why, what wasn't clear from the video, the questions I asked, and the answers I got wrong. The wrong answers taught the most, so they're kept, with what was wrong in them. The step-by-step instructions, settings and losses are in [plan.md](plan.md).

How to read the labels:

| Label | Meaning |
|---|---|
| **Decision** | A choice about how to build it, and the reason |
| **Not clear from the video** | Something the video went past too fast, and the answer |
| **Question** | Something I asked while building |
| **Prediction (right / wrong / half right)** | My answer to a question asked before a run or a check, and what was wrong in it |

Status: all 8 steps done 2026-11-20 (started 2026-11-17). A section is added after each step, before its commit.

## Before building: questions from watching the video

Asked on 2026-11-16, while watching the video up to chapter 21, before deciding to build it (Setup, below). Most answers are expanded in the step where the idea was built, named after each one.

- **Not clear from the video: do I need his separate bigram video?** (makemore part 1.) No, only the idea: a bigram model predicts the next character from the current one alone, with a table of "after X, how likely is each Y." The GPT video starts there because it's the baseline attention has to beat. Watch makemore only if that part feels too fast. *(Step 3.)*
- **Not clear from the video: how does `nn.Embedding(vocab_size, vocab_size)` become a lookup table?** Its weight is the table, and its `forward` is `weight[idx]`: row indexing, no multiplication. *(Step 3.)*
- **Prediction (right): which rows of the table get a gradient?** "The rows matching the values in `xb`." Only those. *(Step 3.)*
- **Question: is `forward` called by `generate`?** Yes, but not by name: `self(idx)` runs `nn.Module.__call__`, which calls `forward`. *(Step 3.)*
- **Question (the wrong way round): is `idx` one batch of 8 blocks?** It's one batch of B blocks, each 8 characters long. *(Step 2.)*
- **Prediction (right): what can't a fully trained bigram learn?** "The context: it only looks at the current character, not the ones before it." *(Step 4.)*
- **Prediction (half right): what does averaging the earlier positions lose?** "Position, order." Right, but it also gives every earlier character the same weight, which is what attention fixes. *(Step 5c.)*
- **Not clear from the video: 58:44 to 59:41, chapter 12** (the token table becomes 65 × 32, with `lm_head` after it). It's a level of indirection: the table gives a 32-number vector, and `lm_head` turns it into 65 scores, so the rest of the Transformer can work on the vectors in between. The `C` he warns about: in the loss code, `C` is 65 (`vocab_size`), not 32 (`n_embd`). Then two predictions, both wrong at first: that `lm_head` alone would lower the loss, and that the position table alone would. *(Steps 5a and 5b.)*
- **Question: how does `tok_emb + pos_emb` add up, with an example?** *(Step 5b.)*
- **Not clear from the video: how can a transpose, `masked_fill` and softmax give a position what it needs from the earlier ones?** They can't on their own: they're fixed plumbing. Key, query and value are learned, and they decide what is looked for and found. *(Step 5c, "How attention learns".)*
- **Question: in Digit Recognizer we still chose the channels; here we don't even do that?** In both you choose how many, never what they mean. *(Step 5c.)*
- **Prediction (half right): why does one head only get to 2.4?** I picked A and C, ruled out B and E correctly, and missed D and F: I wasn't sure why D (no processing after gathering) mattered. *(Step 5c.)*
- **Not clear from the video: residual connections,** which he doesn't explain in detail. First, how a gradient passes through an addition (copied to both inputs), then the prediction about `x + block(x)`: not sure, until drawing the two paths gave **g**. *(Step 7b.)*

## Setup

- **Decision: build it rather than run it.** After the video, my first plan was to run Karpathy's `gpt.py` unchanged on a different text. I chose to build it by hand instead: getting stuck and getting things wrong is what makes the concepts clear. So I write each step from what I understood, without copying from the video, and go back to the video only when stuck. Claude reviews, asks for a prediction before each run, and writes these notes.
- **Decision: one file, `gpt.py`, in Karpathy's order** (settings, data, batches, model, training, generation), so it can be compared with his at any point.
- **Decision: the seed before the model** (`torch.manual_seed(1337)` at the top). The Digit Recognizer lesson: a model built before the seed started from different weights on every run.
- **Decision: one commit per step,** once its checks pass, with the step's section here and its row in plan.md.
- **The global Python has `torch 2.15.0+cpu`, the CPU-only build.** It trains without complaint, just on the CPU and hours slower. Check `torch.cuda.is_available()` in the environment you actually run. This folder's venv has `torch 2.15.0+cu132`.
- **CUDA versions:** the driver supports CUDA 13.3, but PyTorch publishes builds only up to 13.2. A driver runs builds made for older CUDA versions, so cu132 works.
- **Question: did Karpathy use an A100?** Yes: the final model trained for about 15 minutes on an A100 (a data-centre GPU, 40 or 80 GB). The RTX 4070 SUPER's 12 GB is enough for his full settings, just slower (in the end, 13 min 42 s: step 8c).

## Choosing the training text

- **Decision: Karpathy's Tiny Shakespeare first,** so every number can be compared with his. Another text comes after.
- **How the text was chosen:** first Jung's *Man and His Symbols*, from an EPUB. Dropped: the copy came from OceanofPDF, a site that distributes books without the publisher's permission. Then Freud's *The Interpretation of Dreams* (Brill's translation, out of copyright), from Gutenberg's single HTML file, with [extract_text.py](extract_text.py) to strip the header, licence, footnotes and bibliography. Not run yet: Gutenberg was down for maintenance. So Shakespeare first, as above.
- **Question: a `PermissionError` when reading the book.** The path pointed to the EPUB's `OEBPS` folder, not a file. On Windows, `open()` on a folder raises `PermissionError` instead of a clearer error.
- **Question: the BeautifulSoup text includes a header.** `get_text` on the whole page takes everything, including `<title>`. Take the text from the elements you want instead: each `<p>` as one line.

What makes a text good for training:

| | Why |
|---|---|
| About 1 MB or more | Much less and a 10.8M-parameter model memorises it: validation loss rises while training loss keeps falling |
| One style throughout | The last 10% becomes the validation set. An index, a licence or a bibliography at the end makes the validation loss look bad for reasons unrelated to the model |
| A modest character set | Every distinct character gets its own row in the table. Hundreds of rare symbols waste rows |

- **The format:** EPUB beats PDF. An EPUB is a zip of HTML pages, so paragraphs are whole `<p>` elements. A PDF mixes page numbers and running headers into the text and splits words at line breaks. A single Gutenberg HTML file beats both: one file, and the boilerplate is marked (`section.pg-boilerplate`).
- **Copyright:** this repo is public. Training on a text locally is fine, but pushing it would republish it, so `input.txt` and downloaded books are in `.gitignore`.
- **Vocabulary prediction (open):** my first guess was that a book's character set would be the same size as Shakespeare's 65, then 80. Shakespeare's 65 are hand-typed ASCII: newline, space, `!$&',-.3:;?` and the 52 letters, with no digits except `3`. A typeset book adds curly quotes, dashes, digits and accented foreign names. To check when Freud's text is extracted.

## Tensors: dimensions and slicing

- **Question: dimensions, axes and slicing were unclear,** so Claude wrote [tensor_slicing.py](tensor_slicing.py): runnable examples of each, ending with the slices used in gpt.py. Asked during step 6; it's here because every step uses it.

- **Picture a 3-D tensor as sheets stacked front to back.** `torch.arange(24).view(2, 3, 4)` is two 3 × 4 sheets: the front one holds 0 to 11, the one behind it 12 to 23. Dimension 0 is which sheet, dimension 1 which row, dimension 2 which column. The outermost dimension holds both sheets; the innermost is the numbers within a row. Like Java's `int[2][3][4]`. **(My first picture was wrong:** one matrix in the outermost dimension and the other in the innermost. A dimension is a direction you index along, not a container for one matrix.)
- **In the model:** `tok_emb` (32, 8, 32) is 32 sheets (blocks), each 8 rows (positions) of 32 numbers (`n_embd`, the length of each vector). The 65 vectors live in the table (65, 32), a separate tensor; the output holds 256 copies picked from it, `tok_emb[b, t] = table[idx[b, t]]`.
- **One entry per dimension, separated by commas.** An integer removes that dimension (`x[0]` is (3, 4)); a slice keeps it (`x[0:1]` is (1, 3, 4)); `:` alone means all of it. `x[0:2]` on (3, 3, 4) keeps the first two sheets: (2, 3, 4).
- **In a 2-D (2, 3) matrix, dimension 0 is the rows and dimension 1 the columns,** in the shape's order. `sum(dim=0)` gives one sum per column, because the dimension you name is the one that disappears: it adds down through the 2 rows. **(I had it the other way round:** dimension 0 as the columns.)

## Step 1: data and tokenizer

- **Decision: split by position, as in the video** (below), not at random: a random split would put neighbouring characters in both sets.

- **A character-level tokenizer** is two dicts: character → id (`stoi`) and id → character (`itos`). The vocabulary is the sorted unique characters, 65 for Shakespeare.
- **The split is by position:** the first 90% trains and the last 10% validates. So the validation text is a continuous part of the book the model never saw.

## Step 2: `get_batch`

- **Decision: `batch_size = 4`, `block_size = 8` for now,** small enough to print a batch and read it.

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

- **Decision: stay at batch 4 and raise the steps to 10,000,** since 5,000 stopped at 2.89, above the bigram floor (below: steps, not characters, decide how far Adam gets).
- **Decision: commit step 4 before changing any settings,** so each result can be reproduced from its commit.
- **Decision: `EVAL_ITERS` separate from `EVAL_INTERVAL`:** one is how many batches an estimate averages, the other how often it runs.
- **Question: the loss is already returned by `forward`, why not use that?** It's the same loss, averaged:

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

- **Decision: Karpathy's settings for steps 5 to 7** (batch 32, block 8, `lr` 1e-3, 5,000 steps, `n_embd` 32), so every loss compares with a number in the video.
- **Question: was the learning rate the same?** Yes, 1e-3 at this stage (step 4 above: his changes twice in the video).
- **Decision: step 5 in three parts,** 5a, 5b and 5c, so the effect of each piece shows on its own.
- **Decision: a baseline for 5a,** a bigram trained at these settings, to know the floor 5a should land on. At `lr` 1e-3 it was still falling at 2.58 after 5,000 steps, so a second run at 1e-2 found the floor, 2.49.

- **A level of indirection:** a row is now a 32-number description of the character, and `lm_head` turns it into 65 scores. The (B, T, 32) tensor in between is where the rest of the Transformer works: position, attention, feed-forward and blocks all take (B, T, 32) and return it. In Spring terms: an internal object between the database row and the response, with a mapper at the end.
- **Prediction, wrong:** "better, since the linear layer adds weights." It's about the same. Follow one position from `idx` to `logits`: its scores still depend only on the current character, so it's still a bigram table, and a bigram table can't beat the best bigram table. **Weights help only when the model has information they can use.**
- **Counting parameters:** `nn.Linear(32, 65)` has 33 numbers per output score (32 weights plus a bias), times 65 scores, so 2,145. PyTorch stores the weight as (out, in) = (65, 32), plus a separate bias of 65. `nn.Linear` has a bias unless you pass `bias=False`. The total is 2,080 (token table) + 2,145 = 4,225, exactly the bigram's 65 × 65, so the capacity is the same. (I first said 33, then 4,290, which counted the bias twice.)
- **Result:** 2.497, at the bigram's floor (2.49), as predicted. It got there faster than the plain bigram at the same `lr`: `lm_head` is shared by every character, so every step trains it, and its small initial weights start the predictions near uniform.

## Step 5b: position embedding

- **Decision: keep the random chunk starts,** even though position alone then predicts nothing (below).
- **Decision: crop `generate`'s input to the last 8 characters,** after the crash at T = 9 (below).

- **A second table, one row per position** (8 × 32), added to the character's row:

| Character at position | Token row + position row | x |
|---|---|---|
| `t` at 2 | [2, 1, 0] + [0, 0, 0.3] | [2, 1, 0.3] |
| `t` at 0 | [2, 1, 0] + [0.1, 0, 0] | [2.1, 1, 0] |

  (3 numbers instead of 32, made up.) The same character in different places gives different vectors.
- **The lookup is the same as for tokens:** `torch.arange(T)` makes the ids 0 to T−1 (stopping before T, like `range`), and each id picks a row of the table. The ids are row numbers, not the values inside the rows: each row holds 32 learned numbers of its own, random at the start (N(0, 1)) like the token rows. There's no loop: all T rows are looked up in one call. In training T = 8, so the lookup returns the whole table in order. **(I first thought the rows themselves would hold 0, 1, 2 … 7:** `arange` makes the ids, not the values in the rows.)
- **Each position adds its own pattern, number by number:** number c of the token row plus number c of the position row, for c = 0 to 31. It is not one constant (like 0.1) added to all 32 numbers; that was only a made-up example to make the change easy to see.
- **Broadcasting:** `pos_emb` is (T, 32) with no batch dimension, so PyTorch treats it as (1, T, 32). Row t of the position table is added to the token vector at position t **in every block**, not to block t (**I first said "block 0 is added with row 0"**): the 8 position rows are reused across all 32 blocks. In Java terms, `out[b][t][c] = tokTable[ids[b][t]][c] + posTable[t][c]`.
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

- **Decision: scale by `head_size`, not `n_embd`.** Equal here, but not once there are several heads.
- **Decision: two checks before trusting the loss,** the rows of `wei` and no peeking at the future, and then **test the test:** break the mask and see both fail.

### What an average loses

Asked while watching the video. "Position, order" was my answer; I missed the equal weights.

Making each position the average of itself and the earlier ones gives it context, but it's "extremely lossy":

| What the average loses | The fix |
|---|---|
| Order and position: `abc` and `cba` average the same | Position embeddings (5b) |
| Every earlier character counts equally | Self-attention: weights computed from the data (query · key) |

### How attention learns

Asked while watching the video: how can a transpose, `masked_fill` and softmax give a position what it needs from the earlier ones?

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

The same reason Java compares doubles with a tolerance: `0.1 + 0.2 == 0.3` is False. **(Prediction, after nudges:** I had no answer at first for why the two compare differently; then "e^(−inf) is 0" and "the floating-point issue.")

**No peeking at the future:** run a batch, change only the last character of each block, run again. Positions 0 to 6 can't see position 7, so their logits must be identical (`torch.equal(logits_1[:, :-1], logits_2[:, :-1])`); position 7 sees its own new character, so its logits must change in every block.

- **Prediction (right), after a nudge: how many blocks did my first version change?** "1." It tested 1 block of 32. `x_block = xb[:, -1]` is the last column, shape (32,), so `x_block[7]` was block 7's last character only. Both asserts still passed, because they were loose: 31 blocks had identical inputs, and `not torch.equal(...)` passes when any single value differs.
- **The fix:** change the whole column on a copy (`xb2 = xb.clone()`; `xb[:, -1]` is a view, so writing to it changes `xb` itself), with `torch.where(last != 0, last // 2, 1)`. `x if cond else y` works on one number but raises "Boolean value of Tensor with more than one value is ambiguous" on a tensor; `torch.where` picks element by element. Then check the last position block by block: `(logits_1[:, -1] != logits_2[:, -1]).any(dim=-1).all()`.
- **Testing the test:** with the `masked_fill` line removed, the `wei` check fails, and with that check also removed, the no-peeking check fails. A check is only worth having if it fails when the thing it guards is broken.

### Result

Validation **2.4146** (train 2.3970), against 2.50 for 5b and Karpathy's 2.4. The first attention head beats the bigram's floor.

A trained `wei` (the first block of a batch) puts most of each row's weight on the diagonal, the position itself: 0.91, 0.91, 0.71, 0.99, 0.69, 0.97, with a little on the one or two characters before.

**Why the diagonal** (I asked for options and got B by elimination): `x = self.sa_head(x)` replaces x, so `lm_head` sees only the head's weighted mix of values. The current character, the single most useful one (the bigram's 2.5), reaches `lm_head` only if the head picks its own position, so training pushes the diagonal up and spends the little that's left on the previous characters (the gain from 2.50 to 2.41). Not the mask (it only zeroes the future; an untrained row 1 is about 50/50), not the scaling (it divides a whole row by the same number), and not too little training (more steps sharpen what helps). The residual in step 7, `x = x + self.sa(x)`, carries the current character past the head, freeing the head to look at the earlier characters.

### Why one head only takes the loss from 2.5 to 2.4

Asked while watching the video, before building.

I picked A and C, ruled out B and E correctly, and missed D and F.

| Limit | Fix |
|---|---|
| A: one head, so one weighted mix of the past | Multi-head (2.28) |
| C: 8 characters of context | `block_size = 256` |
| D: no processing after gathering | Feed-forward (2.24) |
| F: 32 numbers per position | `n_embd = 384` |

- **Why D matters:** `wei @ v` is a weighted average and `lm_head` is linear, so the model can only add evidence up. It can't express "the character before me was `t` **and** I'm `h`, so next is `e`": that needs a nonlinearity. In the Digit Recognizer, adding one ReLU layer took the logistic baseline's 91.31% to 96.60%. Two linear layers without a ReLU collapse into one.
- **Ruled out:** B (the mask hides the future, but the future isn't available when generating anyway) and E (without the scaling, softmax peaks at the highest score).

## Step 6a: multi-head attention

- **Decision: the same settings as step 5,** so each change compares with 5c's 2.415.
- **Decision: 4 heads of 8,** the same parameters as one head of 32, so any gain comes from the structure.

- **The prediction (right):** 4 heads of 8 have the same parameters as 1 head of 32: 4 × 3 × (32 × 8) = 3 × (32 × 32) = 3,072, so the total stays at 7,553. Any gain comes from the structure, not from more weights (the reverse of 5a: there, more weights without new information gained nothing). What the structure adds is four separate softmaxes, so each position gets four weighted mixes of the past instead of one.
- **`nn.ModuleList`, not a Python list:** a plain list hides the heads from `model.parameters()`, so AdamW would never train them and `.to(DEVICE)` would never move them.
- **The concatenation bug:** `torch.cat(..., dim=1)` gave (32, 32, 8) instead of (32, 8, 32). Each head returns (B, T, 8), and `dim=1` is T, so the four heads' rows were stacked below each other, as if each block had 32 positions. `dim=2` (or `dim=-1`, the last dimension whatever the number before it) puts the four 8-number pieces side by side within each position: 8 + 8 + 8 + 8 = 32. This one failed loudly at `lm_head`, since 8 ≠ 32; a mix-up between two dimensions of equal size would fail silently. (Later I thought these notes gave the fix as `dim=1`. They say `dim=-1`: I'd misread it.)
- **The heads split the work.** Looking at the first block's `wei` for each head:

| Head | Where most of rows 2 to 6 sits | Job, roughly |
|---|---|---|
| 0 | the previous character (0.82, 0.93) | "what came just before me" |
| 1 | the previous character, or two back (0.92) | the same, further back |
| 2 | the diagonal (0.98, 0.89, 0.99, 1.00, 0.90) | carries the current character |
| 3 | split between itself and the previous (0.53 / 0.47) | a mix |

  **My first reading said heads 0 and 1 put most weight "before or after the diagonal."** Never after: the future is masked to exactly 0. This answers 5c's diagonal question from the other side: the current character reaches `lm_head` only if a head passes it on. With one head that took nearly all the weight; with four, one head takes that job and frees the others to look back.
- **Result:** validation **2.2728** (train 2.2459), against 5c's 2.4146 and Karpathy's 2.28.

## Step 6b: feed-forward

- **Decision: an assert that adds up the parameters layer by layer** (mine), instead of only printing the total, so a count that's off by one layer fails the run.

- **Parameters (predicted 32 × 32 at first):** `nn.Linear(32, 32)` has a bias too, so 1,024 + 32 = **1,056**. ReLU is max(0, x) per number and has nothing to learn. The total is 8,609, checked in gpt.py by adding up the layers:

| Layer | Count |
|---|---|
| Token table, 65 × 32 | 2,080 |
| Position table, 8 × 32 | 256 |
| Key, query, value: 4 heads × 3 × (32 × 8) | 3,072 |
| Feed-forward, 32 × 32 + 32 | 1,056 |
| `lm_head`, 32 × 65 + 65 | 2,145 |
| **Total** | **8,609** |

  No parameters: the `tril` buffer, ReLU, softmax, `cat`.
- **Prediction (half right), from options: why the feed-forward layer can't break the no-peeking test.** I chose B, "attention only looked back, so the feed-forward layer can't look forward." Two parts are needed. The mask makes each position's vector hold only its own past; and `nn.Linear` acts on the last dimension only, so position 3's output uses only position 3's vector. "Attention only looked back" alone isn't enough: position 4's vector, right next to it, holds position 4, and a layer that mixed positions could pull it into position 3.
- **Question: how does `lm_head` take the last dimension of (B, T, 4 × head_size)?** How `nn.Linear` handles (B, T, 32): it computes `x @ W.T + b`, (B, T, 32) @ (32, out) + (out,). The last dimension is used up and replaced; B and T are not named, so they're kept, and the same `W` is applied at every (b, t). In the sheets picture, every row of every sheet is transformed on its own. `lm_head` doesn't know its 32 numbers came from 4 heads joined together.
- **Question: `wei` is (B, T, T) and the values (B, T, head_size), so how does `wei @ v` multiply?** A batched matrix multiply: (B, T, T) @ (B, T, head_size). B is matched block by block, and each block is an ordinary (T, T) @ (T, head_size) = (T, head_size): the inner sizes match and disappear. Row t of `wei` holds position t's weights, so row t of the result is the weighted sum of the value vectors: with weights [0.2, 0.3, 0.5] and values [10, 20], [30, 40], [50, 60], it's [36, 46].
- **Question (worked out right): the shape after `sa_heads`.** each head returns (B, T, 8); the concatenation is (B, T, 4 × 8) = (B, T, n_embd) again, because `head_size = n_embd // 4`. That needs `n_embd` to divide evenly: with 30 and 4 heads, 4 × 7 = 28, and `lm_head`'s `Linear(30, 65)` would fail. Karpathy's final settings: 384 = 6 × 64.
- **Result:** validation **2.2342** (train 2.2059), against 6a's 2.2728 and Karpathy's 2.24.

## Step 7a: three blocks, no residuals

- **Decision: blocks without residuals first,** to see the problem residuals solve before adding them.

`Block` = multi-head attention, then feed-forward. `nn.Sequential` chains 3 of them, each block's output going into the next. The checks moved to `model.blocks[0].sa.heads`, and the parameter assert counts the blocks: 3 × (3,072 + 1,056) + 4,481 = **16,865**.

| | 6b | 7a |
|---|---|---|
| Parameters | 8,609 | 16,865 |
| Validation loss | 2.2342 | **2.3787** |
| Training loss | 2.2059 | 2.3708 |

- **Prediction (wrong):** deeper is better, because the extra attention and feed-forward layers get more chances to adjust the weights. The loss went **up**.
- **Not a capacity problem:** 3 blocks can do everything 1 block can (the other two could learn to pass x through). The model is harder to *train*: it was still at 3.13 after 500 steps, and still falling at 5,000.
- **"More steps" is a fix, not the reason:** a 6b-sized model reached 2.23 in the same 5,000 steps. The question is why this one learns more slowly.
- **Backward:** the gradient for the token table is the product of the local gradients of all 6 layers above it (the chain rule). Each one can shrink it, so the embeddings get a small gradient and learn slowly: vanishing gradients, as in RNNs.
- **Forward:** `x = layer(x)` replaces x 6 times. In 5c one replacement already meant the character survived only if the head kept it. Now every layer has to learn to pass it on, and the ReLU zeroes the negative half each time, before the layer above has anything useful to work with.
- Both halves are fixed by residuals (7b): `x = x + layer(x)` keeps x in the stream and gives the gradient a direct path back.

## Step 7b: residuals, `proj`, a wider feed-forward

Three changes: `x = x + self.sa(x)` and `x = x + self.ffwd(x)` in `Block`; `proj = Linear(32, 32)` after the heads' `cat`; the feed-forward becomes 32 → 128 → 32. Per block: 3,072 (heads) + 1,056 (`proj`) + 4,224 + 4,128 (feed-forward) = 12,480, so 3 × 12,480 + 4,481 = **41,921**.

| | 6b | 7a | 7b |
|---|---|---|---|
| Parameters | 8,609 | 16,865 | 41,921 |
| Validation loss | 2.2342 | 2.3787 | **2.0927** |
| Training loss | 2.2059 | 2.3708 | 2.0008 |
| At step 500 (val) | | 3.1313 | 2.3733 |

- **The residuals fixed 7a:** at step 500 the loss is already 2.37 (7a was at 3.13). The same 3 blocks, now with a direct path for x and for the gradient. Karpathy: 2.08.
- **The text has more real words** ("And they", "Whith full him", "you love", "under", "son", "my of"), and character names with a colon on their own line.
- **A gap opens:** train is 0.09 below val (0.03 in 6b). With 41,921 parameters the model starts to memorise the training text. Dropout (step 8) is aimed at this; layer norm (7c) is not.

### Why `proj` is a `Linear`, not a fixed mix

- **The question:** if the 4 heads' outputs sit side by side and `proj` is meant to mix them, why not mix them by hand?
- **We don't know the right mix.** What each head does is only decided during training (in 6a: one kept the current character, two looked back one or two characters, one mixed). A `Linear` is a learned mix: each of its 32 outputs is a weighted sum of all 32 inputs.
- **Fixed mixes lose things:** summing or averaging the 4 pieces gives 8 numbers, not 32, so it can't be added to x, and it blurs what each head found.
- **The real reason is the residual:** without `proj`, `x + cat(heads)` adds head 0's output to channels 0 to 7 of x only, head 1's to 8 to 15, and so on: each head could write to its own quarter of the stream only. With `proj`, every head can write to all 32 channels.

### Not clear from the video: gradients through an addition

Asked while watching the video, as the first half of the residuals question.

At `z = x + y`, the gradient is **copied** to both inputs, not split: ∂z/∂x = ∂z/∂y = 1, so both get the full g. With x = 3, y = 4 and L = z², both get 14. Multiplication instead scales each by the other input. When a value is used in two places, its gradients from both uses are summed.

### Not clear from the video: residual connections

Asked while watching the video: he doesn't explain them in detail.

- **Prediction:** with `x = x + block(x)` and a block that passes almost no gradient back, the incoming x still gets about **g**: the full gradient through the direct path, plus nearly nothing through the block. (I wasn't sure at first; drawing the two paths answered it.)
- **The problem without them:** a chain of blocks multiplies their local gradients. At 0.5 per block, 6 blocks give 0.016 and 20 give 0.000001: the early layers stop learning.
- **The fix:** the local gradient of `x + f(x)` is 1 + f′, so the product always contains 1 × 1 × … × 1, a direct path to the embeddings. Karpathy's "gradient super highway."
- **Why "residual":** the blocks start small, so `x + f(x) ≈ x` and the network trains like a shallow one first. Each block learns a correction to x, not a new x.
- **The residual stream:** x flows from the embeddings to `lm_head`, and each layer reads from it and adds its result back. It also keeps the character's own embedding in the stream: `x = self.sa_head(x)` in 5c replaces it.
- **What it adds to the code:** `proj` (a `Linear` back to `n_embd`, so the heads' output can be added to x), and pre-norm, `x = x + self.sa(self.ln1(x))`, with the layer norm on the branch so the direct path stays a pure addition.
- From ResNet (2015), which trained 152-layer image networks. Karpathy's model has 6 blocks; GPT-3 has 96.

## Step 7c: layer norm

- **Question: how does layer norm work, and how would it help here?** (Below.)
- **Decision: run its check on a fresh model,** since training moves γ and β.

### What it does

`nn.LayerNorm(32)` takes one position's 32 numbers and rescales them to mean 0 and standard deviation 1, then multiplies by a learned γ and adds a learned β (32 each, starting at 1 and 0). With 4 numbers instead of 32:

| Step | Result |
|---|---|
| x | [2, 4, 6, 8] |
| Subtract the mean (5) | [−3, −1, 1, 3] |
| Divide by the std, √((9+1+1+9)/4) = 2.24 | [−1.34, −0.45, 0.45, 1.34] |
| × γ + β | unchanged at the start (γ = 1, β = 0) |

- **Within one position, across the last dimension.** Batch norm instead normalises each channel across the batch. Layer norm doesn't depend on the other blocks in the batch, and it can't mix positions, so the no-peeking test still passes.
- **γ and β** let the model undo the normalisation where that helps, for example if a channel should stay large.

### Why it helps

- **The residual stream grows:** x = x + sa(x) + ffwd(x) + …, every layer adds to it, so its size drifts block by block and during training.
- **The layers reading it have to chase that size.** Attention suffers most: large inputs give large q·k scores, the softmax becomes nearly one-hot, and its gradients become tiny (the reason for `head_size**-0.5` in 5c).
- **Pre-norm:** `x + self.sa(self.ln1(x))`. The norm is on the branch, so each layer gets its input at a fixed size and the stream itself stays a pure addition: the gradient's direct path is untouched. The paper normalised after the addition; GPT moved it before. `ln_f` does the same for `lm_head`.
- **Like converting amounts to one currency** before a service compares them: the service no longer needs to know how big the input came in.
- **A small gain here** (Karpathy: 2.08 → 2.06): 3 blocks don't drift far. It matters with many blocks (GPT-3 has 96).

### Parameters

- **Prediction (half right):** one layer norm before attention, one before the feed-forward, one before `lm_head`: 3. The kinds were right, the count wasn't: `ln1` and `ln2` are inside `Block`, which is built 3 times, so it's 3 × 2 + 1 = **7** layer norms.
- 7 × 64 = 448, so 41,921 + 448 = **42,369**.

### The check

- **Question: the mean over the last dimension gives a tensor with the other dimensions' shape.** Yes, (B, T): that's what to check.
- `ln_f`'s output, stored in `forward` as `self.ln_f_output` (B, T, 32), from a **fresh, untrained** model: after training, γ and β have moved, so the output is no longer mean 0, std 1. A new model built after training leaves the training run unchanged.
- `mean(dim=-1)` gives (B, T): one mean per position, which is what to check, since each position is normalised on its own. `allclose` against zeros (`atol=1e-5`), the same pattern as `wei.sum(dim=-1)` against ones.
- `std(dim=-1, correction=0)`: `std` divides by n − 1 by default, layer norm by n. With n = 32 the default gives about 1.016 and the assert fails.

### Result

| | 7b | 7c |
|---|---|---|
| Parameters | 41,921 | 42,369 |
| Validation loss | 2.0927 | **2.0582** |
| Training loss | 2.0008 | 1.9727 |
| Gap | 0.092 | 0.086 |

- Karpathy: 2.06. All the checks pass: the layer norm, `wei`, the parameter count and no-peeking.
- **The gap is about the same:** layer norm helps the model train, it doesn't stop it memorising. That's dropout's job (step 8).
- Step 7 overall: 2.234 → 2.058, from 3 blocks that only work with residuals.

### Question after 7c: why not one layer norm before the blocks?

- **The question:** normalise x once before `self.blocks`, so every block gets a normalised x, and drop the two per block.
- **The size drifts after every addition, and those are inside the blocks.** `x = x + sa(x)` grows x, then the feed-forward reads the grown x, then block 2, block 3. One norm at the start covers only the first of the 6 places a layer reads the stream.
- **On the stream itself** (`x = ln(x)`), it would also break the pure addition, and with it the gradient's direct path back to the embeddings.
- **The saving is small:** 64 parameters per layer norm, 448 for all seven, out of 42,369.

## Step 8a: hyperparameters instead of fixed numbers

- **Decision: step 8 in three parts,** so the long run starts only once the code is known to be right.
- **Decision: a refactor first, checked by an identical result.**
- **Decision: run outputs go in `output.txt`** (gitignored), and Claude reads them there instead of re-running: each run takes 90 seconds now, and the long one about 14 minutes.

- `N_EMBD`, `N_HEAD = 4`, `N_LAYER = 3` at the top, and the blocks built in a loop: `nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])`. The `*` unpacks the list into separate arguments, like Java's varargs.
- **The check: the same result, exactly.** 42,369 parameters, and every loss line identical to 7c, down to 2.0582. Same seed, same model, built in the same order: a refactor that changes a number has changed the model.
- **Prediction (half right), finding them:** the head size, yes; then I asked whether the `4` in the feed-forward was the other one. It wasn't. **Typed-in numbers that happen to be right:** after the refactor the parameter count still had `N_EMBD * 8` (the head size, right only for 32 / 4) and `layer_norm_weights = 64` (right only for 32). Both were correct today and would have failed the assert at 8c's settings. They became `HEAD_SIZE` and `N_EMBD + N_EMBD`. The `4` in the feed-forward stays: `FeedForward` is written with `4 * n_embd`, so it's part of the design, not of the settings.
- The formula, run on its own with 8c's settings, already gives the count Karpathy prints, before any training.

## Step 8b: dropout, at the small settings

- **Decision: dropout at the small settings first,** to see its effect on its own before scaling up.
- **Decision: `model.eval()` before `generate`.** I first put it before the checks; one line earlier also samples the text with the whole network.

`DROP_OUT = 0.2`, in three places: on `wei` after the softmax, after `proj`, and as the last layer of the feed-forward. In training mode `nn.Dropout` sets a random 20% to 0 and scales the rest by 1 / 0.8 = 1.25, so the expected total stays the same. In eval mode it returns its input unchanged.

### Which checks survive dropout

The model is in training mode after the loop (`estimate_loss` ends with `model.train()`), so dropout is active during the checks.

| Check | With dropout active | Why |
|---|---|---|
| Layer norm | passes | `ln_f` normalises whatever reaches it, zeros included |
| `wei` | **fails** | stored after dropout: some entries 0, the rest × 1.25, so rows don't sum to 1 |
| Parameters | passes | dropout has no parameters |
| No-peeking | **fails** | each forward pass drops a different random 20%, so positions 0 to 6 differ between the two runs even with the same input |

- **I first answered with layers** (token embedding, position, `lm_head`) instead of checks; the table above is the question asked again.
- **I first thought no-peeking would pass:** the weights don't change between the two runs, but which numbers are dropped does (options helped). The test would fail without the mask being broken: it can't tell the two causes apart.
- **The fix is one `model.eval()`,** placed before `generate`, so the text is also sampled with the whole network (it was sampled in training mode before).
- **`eval()` sets a flag, it doesn't run anything:** it sets `self.training = False` on every module, and `nn.Dropout` checks it on each call. I thought the `wei` check passed only because `generate` ran first, but the check runs its own forward pass, `model(xb, yb)`, which overwrites `self.wei`. The mode at that call is all that matters.
- **`nn.Dropout`, not `F.dropout`:** the function form defaults to `training=True` and drops even in eval mode.

### Result

| | 7c | 8b |
|---|---|---|
| Validation loss | 2.0582 | **2.1317** |
| Training loss | 1.9727 | 2.0871 |
| Gap | 0.085 | **0.045** |

- **Prediction (mostly right):** the gap shrinks, because the training loss rises while validation stays about the same. The gap halved and training rose, but validation got **worse** by 0.07.
- **Why it hurt:** 42,369 parameters is small for the text, so the model was short of capacity, not memorising much (a gap of 0.085). Dropout had little memorising to remove, and it switched off 20% of a model that was already too small.
- **Not because the text is small:** a smaller text is easier to memorise, so dropout would help more. What matters is the size of the model against the text: about 24 characters per parameter here, about 0.1 at 8c's 10.8 million. That's why Karpathy adds dropout only when he scales up.
- The reported training loss is measured in eval mode (`estimate_loss`), so it's not raised by dropout noise: the model really fits the training text less well.

## Step 8c: Karpathy's final settings, the long run

- **Decision: time 100 steps first,** to know how long the full run takes before starting it.
- **Decision: print only `wei[0, :8, :8]`;** 256 × 256 is unreadable.
- **Decision: save the weights** to `model.pt` (gitignored), so more text can be generated without another run.

Batch 64, block 256, `lr` 3e-4, 384 wide, 6 heads of 64, 6 blocks, dropout 0.2, 5,000 steps. **10,788,929 parameters**, from the formula of 8a before any training, and the same as Karpathy's.

### How long it takes

- **Prediction (wrong):** 1 minute per 100 steps, so about 50 minutes. The first timing said 31 seconds, but the clock included the step-0 estimate: 400 forward passes against about 300 forward-pass equivalents for 100 training steps (a backward pass costs about two forwards). That gave about 0.13 s per step and an estimate of about 15 minutes.
- **Measured:** 80 seconds per 500 steps, estimate included, and **13 minutes 42 seconds** in all (Karpathy: about 15 on an A100). The small model took about 90 seconds.

### The gap

- **Prediction (right), from reasons: the gap at step 5000,** asked at step 1500 (gap 0.21). I wasn't sure, so I asked for reasons to weigh, right and wrong. I picked A and D, both right, so: bigger.
  - **A (right):** each training character is seen about 80 times over 5,000 steps (64 × 256 per step), so the model keeps fitting the exact text it keeps seeing.
  - **B (wrong):** "dropout stops memorising." It slows it down.
  - **C (half right):** real patterns help both losses, but memorising grows at the same time and only helps the training loss.
  - **D (right):** 10.8 million parameters for 1 million training characters is room to store the text.

| Step | Train | Val | Gap |
|---|---|---|---|
| 500 | 1.8937 | 2.0042 | 0.11 |
| 1500 | 1.3955 | 1.6072 | 0.21 |
| 3000 | 1.2021 | 1.4993 | 0.30 |
| 3500 | 1.1602 | 1.4899 | 0.33 |
| 4500 | 1.0879 | **1.4856** | 0.40 |
| 5000 | 1.0546 | 1.4994 | **0.45** |

- **Validation stopped improving around step 3000** (1.499, then 1.490, 1.491, 1.486, 1.499), while training kept falling by about 0.04 per 500 steps. The last 2,000 steps were mostly memorising. The best validation, 1.4856 at step 4500, matches Karpathy's 1.48; the final weights are from step 5000 (1.4994).
- Saving the weights at the best validation (early stopping, as in Digit Recognizer) would have kept 1.4856.

### The text

Reads like Shakespeare at a distance: names and line breaks, real words, the rhythm of verse, and sentences that mean nothing:

```
Where thy prince. Come, we may; this last
She old Clarence, and hagg's virginicly study:
Commend them and Barnardine's fall,
Thou mayst do it, I will speak no foot;
```

From 8 characters of context and 42,369 parameters (2.06) to 256 characters and 10.8 million (1.49).

### The checks at full size

All pass at T = 256: the layer norm on a fresh model, `wei` (printed as an 8 × 8 corner, `wei[0, :8, :8]`), the parameter count, and no-peeking (positions 0 to 254 identical when only character 255 changes). The trained weights are in `model.pt` (52 MB, gitignored), so more text can be generated without another run.

## After step 8: training behind `__main__`

- **Decision: move training, generation and the checks into functions** (`train_and_generate()`, `checks_and_validations(model)`) called under `if __name__ == "__main__":`. Without the guard, importing the model class from gpt.py (for example, to load `model.pt` and generate more text) would run the whole 14-minute training.
- **The rerun gave 1.4958 (train 1.0543) against 1.4994 (train 1.0546).** Not the refactor: the seed still runs at import and the random calls come in the same order. With the same seed, CUDA doesn't promise bit-identical runs unless deterministic algorithms are turned on, and at 64 × 256 positions per batch tiny rounding differences grow over 5,000 steps. A difference of 0.004 is within the validation loss's own movement over the last 2,000 steps (1.486 to 1.499). At the small settings, reruns had matched exactly.

## The whole build

| Step | Validation loss | Parameters | What changed |
|---|---|---|---|
| 4 | 2.54 | 4,225 | Bigram, trained |
| 5c | 2.415 | 7,553 | One attention head |
| 6a | 2.273 | 7,553 | 4 heads of 8 |
| 6b | 2.234 | 8,609 | Feed-forward |
| 7a | 2.379 | 16,865 | 3 blocks, no residuals: worse |
| 7b | 2.093 | 41,921 | Residuals, `proj`, a 4× feed-forward |
| 7c | 2.058 | 42,369 | Layer norm |
| 8b | 2.132 | 42,369 | Dropout at the small size: worse |
| 8c | 1.499 (best 1.486) | 10,788,929 | Karpathy's settings |
