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

## CNN (decided 2026-11-05)

`build_conv_network` (written by the user): `nn.Unflatten(1, (1, 28, 28))` (the 784 pixels back into a one-channel 28 × 28 grid), then two blocks of `Conv2d` (3 × 3, padding 1, 32 then 64 filters) + ReLU + `MaxPool2d(2)` (28 → 14 → 7), then `Flatten` (64 × 7 × 7 = 3,136), `Linear(3136, 128)`, ReLU, `Linear(128, 10)`. Filters look at small neighbourhoods, so local shape (a closed or open top on a 4 or 9) is something it can learn; the dense network sees every pixel separately. A first run with early stopping gave 98.96% (best epoch 24).

**Result** (`compare_networks` in `data_analysis.py`: the reference setup, 10 epochs on all 33,600, last epoch kept, seeds 0 to 4, so only the network differs):

| Seed | Dense | CNN |
|---|---|---|
| 0 | 96.67% | 98.82% |
| 1 | 96.40% | 98.60% |
| 2 | 96.54% | 98.52% |
| 3 | 96.49% | 98.69% |
| 4 | 96.63% | 98.64% |
| **Mean** | **96.55%** | **98.65%** |
| Sd | 0.11 | 0.11 |

**Decision: kept.** +2.11 points against the +0.5 rule (≥ 97.05%), and every CNN seed beats every dense seed. Mistakes fall from about 290 to about 113 of 8,400. The dense means match the earlier comparison exactly, so seeded runs now reproduce. The 10 runs together took 41 seconds.

## Default training: early stopping (decided 2026-11-05)

**Change of procedure (the user):** from now on every network trains up to 30 epochs and keeps the epoch with the best early-stopping accuracy (`train_network`'s defaults; `compare_networks` uses them unless told otherwise). **Reason, independent of the +0.38 result:** a fixed epoch count chosen for one network doesn't carry over to the next (the CNN may need a different number than the dense network), while the early-stopping set checks every network itself. The cost: each network trains on 26,880 images instead of 33,600. The CNN decision above stands (it was made under the old setup).

The CNN under the new default (seeds 0 to 4): held-out 98.81%, 98.85%, 98.80%, 98.89%, 98.76%, **mean 98.82%** (sd 0.05), against 98.65% with 10 epochs. Best epochs 26, 27, 28, 28, 30: close to the 30 limit, so a longer run might gain a little more.

**Reproducibility fix (2026-11-05):** CNN runs with the same seed still differed (98.11% and 98.02% after 3 epochs; the dense network repeats exactly). Cause: cuDNN, the GPU library behind `Conv2d`, picks fast algorithms that add in a varying order. `train_network` now sets `torch.backends.cudnn.deterministic = True` and `benchmark = False`; same-seed CNN runs are identical again. The CNN under the default, rerun deterministically (seeds 0 to 4): 98.73%, 98.92%, 98.85%, 98.83%, 98.89%, **mean 98.84%**, best epochs 23 to 30. (The 10-epoch CNN numbers above came from non-deterministic runs; the decision doesn't depend on that.)

## Kaggle submission 1 (2026-11-05)

**Training data (option b, the user):** no held-out set for the submission; the CNN trains on all 42,000 labelled images, 33,600 to train on and 8,400 to choose the epoch (best of 30, seed 42). `create_submission()` in `digit_recognizer.py`. Best epoch 25 (early-stopping accuracy 99.19%). Predicted shares per digit 9.0% (fives) to 11.3% (ones), matching the training labels.

**Expected score (before submitting):** about 98.8% to 99.0%: 98.84% was measured on 26,880 training images, and 33,600 should help slightly.

**Kaggle score: 98.960%** (public), inside the expected 98.8% to 99.0%. The held-out estimate carried over to Kaggle's test images.

## Keep rule near 99% (decided 2026-11-06)

The +0.5 rule (about 3 × 0.15, set for a single model's score at 98%) is too strict near 99%: it would need about 40 of the CNN's ~97 mistakes fixed. Recomputed noise, for a **difference** between two models judged on the same 8,400 images:

- **Images:** only the images where the two models disagree change the difference, so its sd ≈ √(disagreements) ÷ 8,400. Measured: two CNN runs that differ only in seed disagree on 60 to 85 images (average ~69; each gets 91 to 107 wrong), so sd ≈ √69 ÷ 8,400 ≈ 0.10 points. (A single model's score has sd √(p(1 − p) ÷ n) = √(0.9884 × 0.0116 ÷ 8,400) ≈ 0.12.)
- **Training:** sd over seeds 0.07 per run (98.73, 98.92, 98.85, 98.83, 98.89), 0.03 for a 5-seed average, √2 × 0.03 ≈ 0.05 for a difference of two averages.
- **Together:** √(0.10² + 0.05²) ≈ 0.11 points. (Independent noises add as variances, not sds.)

**Rule (the user):** a change is kept if its 5-seed average held-out accuracy beats the current model's by at least **+0.3 points** (about 2.7 sd; about 25 fewer mistakes). **Check:** count the disagreements in each comparison; if they're well above 69, recompute the sd before deciding.

Observation for later: seeds 0 and 1 share 65 mistakes (both wrong); seed 0 has 42 of its own, seed 1 26. So about 60% of a run's mistakes are shared (hard images) and 30 to 40% are specific to the seed (the other run gets them right). Averaging the predictions of several seeds could fix the seed-specific ones, not the shared ones.

## Averaging seeds (started 2026-11-06)

Idea: combine 5 CNNs trained with different seeds; the 30 to 40% of mistakes that are seed-specific can be outvoted by the networks that get them right, the shared ones can't.

**Combining method:** votes (the user's first choice: a network wrong because of its seed is outvoted by the others) or averaged probabilities (a confident network counts more). Each can be shown right with one example (a 3–2 vote of hesitant networks against two confident ones goes either way depending on the true digit), so it's settled by data: which is right more often when the two disagree.

**Decision rule (before running):** on the 6,720 early-stopping images (the same for every seed; the held-out 8,400 stay for the final judgement), 5 CNNs (seeds 0 to 4): the method with more correct images wins; a tie goes to votes. The networks chose their epochs on this set, so their scores on it are a little optimistic, but equally for both methods. `compare_combinations()` in `data_analysis.py`.

**Result (early-stopping set, seeds 0 to 4):** single networks 98.87%, 98.91%, 98.93%, 98.91%, 99.05% (mean 98.93%, about 72 wrong). Votes **99.08%** (62 wrong, 2 ties), averaged probabilities **99.08%** (62 wrong). They differ on only 6 images, 3 right for each. **Decision: votes** (a tie, so the rule's tie-break). The two methods hardly ever disagree: when the networks disagree, a hesitant majority against confident ones is rare.

**Held-out judgement:** `judge_voting()`: 5 voting models (seeds 0–4, 5–9, …, 20–24), against the single CNN's 98.84% (seeds 0–4); kept if the mean is at least 99.14%. **Prediction (the user, before running): it won't clear the rule.** On the early-stopping set the gain was +0.15 (98.93% → 99.08%), half the +0.3 needed.

**Result (held-out set):**

| Seeds | Voting | Ties | First seed alone | Disagreements with it |
|---|---|---|---|---|
| 0–4 | 99.05% | 8 | 98.73% | 47 |
| 5–9 | 99.12% | 6 | 99.04% | 31 |
| 10–14 | 99.10% | 5 | 98.92% | 43 |
| 15–19 | 99.07% | 11 | 98.86% | 48 |
| 20–24 | 99.02% | 8 | 98.90% | 30 |
| **Mean** | **99.07%** (sd 0.04) | | | |

**Decision: not kept; the single CNN stays.** +0.23 points against the +0.3 rule (99.14%); the prediction held. Disagreements with a single CNN (30 to 48) are below the ~69 the rule's sd assumed, so the noise check doesn't change the bar. Voting costs 5× the training for about 19 fewer mistakes of 8,400. `judge_voting()` in `data_analysis.py`.
