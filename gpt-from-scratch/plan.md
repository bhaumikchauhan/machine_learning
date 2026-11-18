# GPT from scratch: the plan

Build a character-level GPT by hand, following Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out", to understand each piece by writing it. Each step is written from what was understood, without copying from the video; the video is for when stuck. Each step is run, checked and reviewed before the next.

- **Text:** Tiny Shakespeare (`input.txt`, 1,115,394 characters), the same as the video, so every number can be compared with Karpathy's. Freud's *The Interpretation of Dreams* comes after, with [extract_text.py](extract_text.py).
- **Hardware:** RTX 4070 SUPER (12 GB), PyTorch with CUDA ([requirements.txt](requirements.txt)). Use this folder's venv: the global Python has the CPU-only build.
- **Layout:** one file, `gpt.py`, in Karpathy's order: hyperparameters at the top, then data, batches, model, training, generation.
- **Seed:** `torch.manual_seed(1337)` at the top, before the model is built (the Digit Recognizer lesson: a model built before the seed starts from different weights every run).
- **Device:** put `.to(device)` on both the model and the data. One on the GPU and the other on the CPU raises a confusing error.
- **Commits:** one per step, once its check passes, with the step's section in [learning_notes.md](learning_notes.md) (what it taught, and the predictions that went wrong).

## Step 1: data and tokenizer

1. Read `input.txt` as UTF-8.
2. `chars` = the sorted unique characters; `vocab_size = len(chars)`.
3. Two dicts, character → id and id → character, then `encode(str) -> list[int]` and `decode(list[int]) -> str`.
4. `data` = the whole text encoded, as a `torch.long` tensor.
5. The first 90% is `train_data`, the rest `val_data`.

**Check:** `vocab_size` is 65, `len(data)` is 1,115,394, and `decode(encode("hii there")) == "hii there"`.

## Step 2: `get_batch(split)`

1. `batch_size = 4` and `block_size = 8` for now, small enough to read.
2. `batch_size` random start offsets, so that `start + block_size + 1` stays inside the data.
3. `x` = the chunks starting at each offset, stacked; `y` = the same chunks shifted one to the right.
4. Both moved to `device` (`'cuda'` when available).

**Check:** both are 4 × 8, `torch.equal(x[:, 1:], y[:, :-1])` is True, and `decode` of a row of x and of y read as the same text shifted by one character.

## Step 3: the bigram model

1. An `nn.Module` with `nn.Embedding(vocab_size, vocab_size)`.
2. `forward(idx, targets=None)` returns `(logits, loss)`: reshape for `F.cross_entropy` (B·T rows); `loss = None` without targets.
3. `generate(idx, max_new_tokens)`: in a loop, take the last time step's logits, softmax, `torch.multinomial`, append.

**Check:** the logits are (4, 8, 65); the first loss is near −ln(1/65) ≈ 4.17 (Karpathy's random start gave 4.87); 100 characters generated from a newline are garbage.

## Step 4: training

1. AdamW with `lr=1e-3`, and a loop: get a batch, forward, `zero_grad`, `backward`, `step`.
2. `estimate_loss()`: the average loss over `eval_iters` batches for train and val, under `@torch.no_grad()`, with `model.eval()` before and `model.train()` after.
3. Print both losses every `eval_interval` steps; generate 500 characters at the end.

**Check:** the loss settles near 2.5; the output has word-like shapes, spaces and line breaks.

## Step 5: embeddings, position, one attention head

**Settings first:** to compare with Karpathy's 2.4, use his settings for this stage: `batch_size = 32`, `block_size = 8`, `lr = 1e-3`, 5,000 steps, and a new `n_embd = 32`. Commit the old settings' result (step 4) before changing them, so each run is reproducible.

### 5a. Embedding size and `lm_head`

1. The token table becomes `nn.Embedding(vocab_size, n_embd)`: a row is now a 32-number description of the character, not its scores.
2. A new `self.lm_head = nn.Linear(n_embd, vocab_size)` turns each 32-number vector into 65 scores.
3. `forward`: `tok_emb` (B, T, n_embd), then `logits = self.lm_head(tok_emb)` (B, T, vocab_size).

**Check:** the loss lands about where the bigram did (your prediction from the video discussion: still only the current character).

### 5b. Position embedding

1. `self.position_embedding_table = nn.Embedding(block_size, n_embd)`: one row per position.
2. `forward`: `pos_emb` = that table looked up with `torch.arange(T, device=device)` (T, n_embd); `x = tok_emb + pos_emb` (broadcast over the batch); `logits = self.lm_head(x)`.

**Check:** the loss is still about the same (random chunk starts: position alone predicts nothing). Generating 500 characters still works. If it doesn't, read the error: it says which table ran out of rows, and why.

### 5c. One self-attention head

1. `class Head(nn.Module)` with `head_size`:
   - `key`, `query`, `value`: each `nn.Linear(n_embd, head_size, bias=False)`.
   - `self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))`: a buffer, not a parameter, so the optimizer never changes it, and `.to(device)` still moves it.
2. `forward(x)`, with x (B, T, n_embd):
   - `k`, `q`: (B, T, head_size).
   - `wei = q @ k.transpose(-2, -1) * head_size**-0.5`: (B, T, T). Scale by `head_size`, not by `n_embd`: they're equal here, but not once there are several heads (step 6).
   - `masked_fill` where `tril[:T, :T] == 0` with `-inf`, then `softmax(dim=-1)`.
   - `out = wei @ self.value(x)`: (B, T, head_size).
3. In the model: `self.sa_head = Head(n_embd)`; in `forward`: `x = self.sa_head(tok_emb + pos_emb)`, then `lm_head`.

**Check:**
- **Shapes and weights:** for one batch, each row of `wei` sums to 1, and everything above the diagonal is 0.
- **No peeking at the future:** run the model on a batch, change only the last character of each row, run it again. The logits at every position except the last must be identical (`torch.equal`). If they change, the mask isn't working.
- **The loss:** about 2.4, below the bigram's 2.5.

## Step 6: multi-head attention, then feed-forward

**Settings:** the same as step 5 (batch 32, block 8, `lr=1e-3`, 5,000 steps, `n_embd = 32`), so each change is compared with 5c's 2.415.

### 6a. Multi-head attention

1. `class MultiHeadAttention(nn.Module)` with `num_heads` and `head_size`:
   - `self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])`. An `nn.ModuleList`, not a plain Python list: a plain list hides the heads from `model.parameters()`, so the optimizer would never train them and `.to(DEVICE)` would never move them.
   - `forward(x)`: run every head on the same x and concatenate their outputs along the last dimension, `torch.cat([h(x) for h in self.heads], dim=-1)`.
2. In the model: `self.sa_heads = MultiHeadAttention(4, n_embd // 4)`: 4 heads of 8, which concatenate back to 32, so `lm_head` stays `Linear(n_embd, vocab_size)`.
3. Each head scales by its own `head_size` (8), not by `n_embd` (32). Yours already uses `k.shape[-1]`, which is right.

**Check:**
- **Shapes:** each head returns (B, T, 8); the concatenation is (B, T, 32).
- **Parameters:** before running, work out the count and compare it with 5c's 7,553. Then print it.
- **The 5c checks still pass,** updated for the new names: the `wei` check for every head, and the no-peeking test unchanged.
- **Look at the heads:** print the first block's `wei` for each of the 4 heads. Do they all attend the same way?
- **The loss:** about 2.28.

### 6b. Feed-forward

1. `class FeedForward(nn.Module)` with `n_embd`: `self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())`.
2. In the model's `forward`: `x = self.sa_heads(x)`, then `x = self.ffwd(x)`, then `lm_head`.
3. It runs on each position separately: the `Linear` acts on the last dimension only, so position t's output depends only on position t's input. Attention gathers from the other positions; the feed-forward layer then works on what each position gathered.

**Check:**
- **Parameters:** work out how many the layer adds, then print the total.
- **The no-peeking test still passes.** Think about why it must, before running it.
- **The loss:** about 2.24.

## Steps 5 to 8 at a glance

| # | Add | Karpathy's validation loss |
|---|---|---|
| 5 | `n_embd` + `lm_head`, position embedding, one `Head` | about 2.4 |
| 6 | Multi-head, then feed-forward | 2.28, then 2.24 |
| 7 | `Block`, residuals with `proj`, layer norm | 2.08, then 2.06 |
| 8 | Scale up to his final settings, dropout, the long run | about 1.48 |

The losses won't match his exactly: different hardware gives different random numbers even with the same seed. They should land close.

## Progress

| Step | Done | Validation loss | Notes |
|---|---|---|---|
| 1 | ✅ | | train 1,003,854 / val 111,540 |
| 2 | ✅ | | randint's upper bound is exclusive: pass `len(data) - block_size` |
| 3 | ✅ | 5.04 (untrained, one batch) | cross_entropy takes raw logits and applies log_softmax itself |
| 4 | ✅ | 2.54 (train 2.54) | Batch 4, `lr=1e-3`: 5,000 steps gave 2.80, 10,000 gave 2.54, the bigram floor. With Adam each step moves a weight by about `lr` whatever the batch size, so the number of steps decides how far the table gets |
| 5a | ✅ | 2.497 (train 2.475) | Batch 32, 5,000 steps, `lr=1e-3`. The bigram's floor with these settings is about 2.49 (trained with `lr=1e-2`; at `lr=1e-3` it was still at 2.58 after 5,000 steps). Same floor, as predicted, but reached faster: `lm_head` is shared by every character, and its small initial weights start the predictions near uniform |
| 5b | ✅ | 2.503 (train 2.484) | 4,481 parameters (+256 for the 8 × 32 position table). Same as 5a within noise, as predicted: chunks start at random, so a position says nothing about which character comes next. `generate` crashed at T = 9 (the table has rows 0 to 7 only), fixed by passing `idx[:, -block_size:]` to `forward` while `idx` itself keeps growing |
| 5c | ✅ | 2.415 (train 2.397) | 7,553 parameters (+3 × 32 × 32 for key, query, value). Both checks pass: rows of `wei` sum to 1 with zeros above the diagonal, and changing each block's last character leaves the logits at positions 0 to 6 identical. Both fail with the mask removed. A trained `wei` puts most weight on the diagonal |
| 6a | | | |
| 6b | | | |
| 7 | | | |
| 8 | | | |
