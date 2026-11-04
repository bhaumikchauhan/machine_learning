# Digit Recognizer notes

Decisions and reasoning for Kaggle's Digit Recognizer: classify 28×28 images of handwritten digits (0 to 9). The first deep learning problem, with PyTorch. Scored by accuracy on Kaggle.

Status: in progress (started 2026-11-04).

**Who writes what** (the same split as before):
- **The user writes:** the network, the training loop and the validation split.
- **Claude writes:** plots, analysis scripts and setup.

Data: `kaggle/input/digit-recognizer/` (not in git). `train.csv`: 42,000 images, a `label` column plus 784 pixel columns (28 × 28, row by row), values 0 to 255, 81% of pixels are 0, no missing values. `test.csv`: 28,000 images without labels. The ten digits are balanced: 9.0% (fives) to 11.2% (ones).

Setup: PyTorch with CUDA, GPU NVIDIA GeForce RTX 4070 SUPER (12 GB).
