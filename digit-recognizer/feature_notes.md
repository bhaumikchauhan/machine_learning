# Digit Recognizer notes

Decisions and reasoning for Kaggle's Digit Recognizer: classify 28×28 images of handwritten digits (0 to 9). The first deep learning problem, with PyTorch. Scored by accuracy on Kaggle.

Status: in progress (started 2026-11-04).

**Who writes what** (the same split as before):
- **The user writes:** the network, the training loop and the validation split.
- **Claude writes:** plots, analysis scripts and setup.

Data: `kaggle/input/digit-recognizer/` (not in git). `train.csv`: 42,000 images, a `label` column plus 784 pixel columns (28 × 28, row by row), values 0 to 255, 81% of pixels are 0, no missing values. `test.csv`: 28,000 images without labels. The ten digits are balanced: 9.0% (fives) to 11.2% (ones).

Setup: PyTorch with CUDA, GPU NVIDIA GeForce RTX 4070 SUPER (12 GB).

## Scaling (decided 2026-11-04)

The gradient contains the input, so raw pixels (up to 255) make every step 255 times bigger than inputs near 1: a learning rate that works for 0–1 inputs would bounce and diverge.

**Decision: divide every pixel by 255** (0 to 1, one shared rule). StandardScaler per pixel column was rejected: a pixel that is almost always blank gets a tiny sd, so a rare stray mark becomes a huge input. pixel14 (top edge) has ink in 1 of 42,000 images; that 216 becomes 204.9 sd, while a full stroke in the centre (pixel406, ink in 66% of images) becomes 1.1 sd. 76 pixel columns are blank in every image.

## Validation (decided 2026-11-04)

**One held-out split: 20% (8,400 images), stratified by label**, fixed seed so every model is judged on the same images. Repeated k-fold (50 fits per comparison) is too slow for networks that take minutes each.

The cost: one split has no fold-to-fold spread to show its noise. Its uncertainty comes from the 8,400 images instead: at 98% accuracy, sd = √(0.98 × 0.02 / 8,400) ≈ 0.15 percentage points.

**Keep rule:** a change is kept if validation accuracy rises by **at least 0.5 percentage points** (about 3 sd of one split's noise), on the same held-out images. Otherwise the simpler model stays.

What the 0.5 left out (noticed 2026-11-05): it covers only which images land in the held-out set. A network also varies with its training (random starting weights, batch order): sd 0.11–0.14 points per run, single runs up to 0.3 apart. Logistic regression has no such noise, so 0.5 was enough for it; for networks, compare 5-seed averages (training noise of the mean ≈ 0.05), where 0.5 holds again.

## Baseline: logistic regression on the 784 pixels

One weight per pixel for each digit (a 28 × 28 template per digit), no notion of which pixels are neighbours. `LogisticRegression(max_iter=1000)` on pixels ÷ 255, judged on the held-out 20%.

**Prediction (before running):** the most confused pair is 1 and 9 (both are mostly one vertical stroke).

**Result:** validation accuracy **91.31%** (730 mistakes of 8,400). `plots/baseline_mistakes.png` (`data_analysis.py`).

| Most common mistakes (true → predicted) | Count |
|---|---|
| 4 → 9 | 42 |
| 3 → 5 | 35 |
| 7 → 9 | 32 |
| 5 → 3 | 32 |
| 9 → 7 | 29 |
| 3 → 2 | 27 |
| 8 → 5 | 25 |
| 8 → 3 | 24 |

The prediction (1 and 9) missed: only 5 mistakes between them (1 → 9: 1, 9 → 1: 4). Hardest digits: 8 (85.4%), 5 (85.5%), 3 (87.6%); easiest: 1 (97.5%), 6 (96.3%), 0 (96.0%).

Why 4 and 9 (and 3 and 5) are confused: not the amount of ink but where it sits. Overlap of the average images (1 = identical): 4 and 9 0.93 (65 mistakes between them), 3 and 5 0.90 (67); 3 and 6 have the same ink (111 and 109) but overlap 0.71 (4 mistakes); 1 and 9 0.62 (5). A template can't tell a closed-top 4 from a 9; that needs local shape.

## Dense network

`nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))`: 128 hidden units, each a partial template, combined by the output layer; ReLU lets a unit switch off (without it the two layers collapse into one, the logistic regression again). The training loop (`train_network` in `digit_recognizer.py`, written by Claude on request): Adam, learning rate 0.001, cross-entropy loss, mini-batches of 128, 10 epochs, fixed seed, the same held-out split (`split_data`, now shared with the baseline). Judged on validation accuracy after epoch 10.

**Prediction (before running):** clears the keep rule (≥ 91.81%). The 128 hidden features can learn strokes and gaps, and the output layer can combine them, which ten fixed templates can't.

**Result:** validation accuracy **96.60%** after epoch 10, against 91.31% for the baseline: +5.29 points, so it's **kept**. The prediction held. Mistakes fell from 730 to 286 (`plots/dense_mistakes.png`).

| Epoch | Train loss | Validation accuracy |
|---|---|---|
| 1 | 0.5225 | 91.58% |
| 2 | 0.2460 | 92.92% |
| 3 | 0.1881 | 94.60% |
| 5 | 0.1245 | 95.57% |
| 7 | 0.0876 | 96.20% |
| 9 | 0.0651 | 96.51% |
| 10 | 0.0570 | 96.60% |

| Mistake (true → predicted) | Baseline | Dense |
|---|---|---|
| 4 → 9 | 42 | 17 |
| 3 → 5 | 35 | 17 |
| 7 → 9 | 32 | 11 |
| 5 → 3 | 32 | 10 |
| 9 → 7 | 29 | 9 |

The top pairs are the same, each cut by half to two thirds. After one epoch the network already matches the baseline. Validation accuracy is still rising at epoch 10 (+0.09 points), but more slowly.

## Number of epochs (decided 2026-11-04)

More epochs keep lowering the training loss; at some point validation accuracy stops rising and falls (overfitting). Picking the best of 30 epochs on the held-out 8,400 would make its score the best of 30 looks, so slightly optimistic.

**Decision (option a):** the 33,600 training images are split again, 80% to train on (26,880) and 20% as an **early-stopping set** (6,720, stratified, seed 42). Train up to 30 epochs; the epoch with the best early-stopping accuracy is kept (its weights restored). The held-out 8,400 are used once, to judge that final model with the keep rule. Same idea as the inner folds of LogisticRegressionCV: choose inside the training data, judge outside it.

**Prediction (before running):** it won't clear the keep rule (≥ 97.10%). At epoch 10 the gain was already small (+0.09 points per epoch) and shrinking, and the network now trains on 6,720 fewer images.

**Result:** three runs of the same code gave three answers. The cause: the network was built (`build_dense_network()` in main) *before* `train_network` set `torch.manual_seed(42)`, so each run started from different random weights; the seed only fixed the batch order. (Claude first blamed GPU rounding; that was wrong. With the network built after the seed, two runs with the same seed give identical results.)

| Run | Best epoch | Early-stopping accuracy | Held-out accuracy | Against 96.60% |
|---|---|---|---|---|
| user's | 29 | 96.83% | 96.74% | +0.14 |
| Claude's 1st | 29 | 97.05% | 97.04% | +0.44 |
| Claude's 2nd | 26 | 96.90% | 97.00% | +0.40 |

None clears the keep rule (+0.5), so the prediction held. The run-to-run spread (0.30 points) is about twice the image-sampling sd (0.15). Fit accuracy reached 100% by epoch 27 while early-stopping accuracy flattened around 96.7–97.0% from epoch 17: the network memorises the training images, but the extra epochs don't hurt it either.

The 96.60% is one draw from random starting weights too, so a +0.4 gain from single runs can't be trusted. The minimum over runs was considered and rejected: it falls as the number of runs grows and rests on the single most extreme run.

**Decision rule (before running):** 5 runs of each version, seeds 0 to 4 (`train_network` now takes `build_network` and `seed`, and builds the network after seeding; `early_stopping=False` trains on all 33,600 and keeps the last epoch). Early stopping is kept if its **average** held-out accuracy beats the 10-epoch average by at least 0.5 points. `compare_epoch_choices()` in `data_analysis.py`.

**Result (5 seeds):**

| Version | Mean held-out accuracy | Sd over seeds |
|---|---|---|
| 10 epochs, all 33,600 | 96.55% | 0.11 |
| Early stopping, best of 30 | 96.93% | 0.14 |
| Gain | +0.38 | |

**Decision: not kept; 10 epochs stays** (96.55% on average is the dense network's score). The gain is several times the noise of a 5-seed average (about 0.05), so it's probably real, but it's below the rule written beforehand. Lowering the rule after seeing a near miss was considered and rejected: a rule moved to fit the result is fitted to the held-out images and gives false confidence.

## CNN: keep rule (decided 2026-11-05)

A keep rule is checked against two things known before running. **The noise sets the minimum:** training noise of a 5-seed average ≈ 0.05 points, but all seeds reuse the same 8,400 held-out images, so the image part (≈ 0.15) stays; the minimum is about 0.5. **What's worth it sets the maximum:** the gain counted in mistakes against the cost of the change. A rule of +2 points would pass the noise check but demand 168 fewer mistakes (58% of the dense network's 290), rejecting a clear improvement. The CNN costs little more than the dense network (a few lines of code, a few minutes of training), so the noise minimum is the bar.

**Rule:** the CNN is kept if its average held-out accuracy over seeds 0 to 4 beats the dense network's 96.55% by at least 0.5 points (≥ 97.05%).
