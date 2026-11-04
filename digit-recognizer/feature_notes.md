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
