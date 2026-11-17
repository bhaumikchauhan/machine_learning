# GPT from scratch: the plan

Build a character-level GPT by hand, following Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out", to understand each piece by writing it. Each step is written from what was understood, without copying from the video; the video is for when stuck. Each step is run, checked and reviewed before the next.

- **Text:** Tiny Shakespeare (`input.txt`, 1,115,394 characters), the same as the video, so every number can be compared with Karpathy's. Freud's *The Interpretation of Dreams* comes after, with [extract_text.py](extract_text.py).
- **Hardware:** RTX 4070 SUPER (12 GB), PyTorch with CUDA ([requirements.txt](requirements.txt)). Use this folder's venv: the global Python has the CPU-only build.
- **Layout:** one file, `gpt.py`, in Karpathy's order: hyperparameters at the top, then data, batches, model, training, generation.
- **Seed:** `torch.manual_seed(1337)` at the top, before the model is built (the Digit Recognizer lesson: a model built before the seed starts from different weights every run).
- **Device:** put `.to(device)` on both the model and the data. One on the GPU and the other on the CPU raises a confusing error.
- **Commits:** one per step, once its check passes.

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

## Steps 5 to 8 (details when we get there)

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
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
