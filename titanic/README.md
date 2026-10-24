# Titanic

Kaggle's [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). Final model: logistic regression on 11 hand-built features plus a group-fate feature (what happened to the women and children on the same ticket). Cross-validation 0.841, local test set 0.838, Kaggle 0.78708.

All decisions, with the data behind them, are in [feature_notes.md](feature_notes.md).

## Setup

Download the competition data from Kaggle into `kaggle/input/competitions/titanic/` (`train.csv`, `test.csv`). Then, from this folder:

```bash
uv run python titanic_v2.py      # v2 (final): cross-validation with and without the group-fate feature
uv run python titanic.py         # v1: the 11 features
uv run python error_analysis.py  # error analysis graphs in plots/
uv run python tree_analysis.py   # decision tree and random forest validation curves
```

## Files

- `common.py`: shared code (data loading, feature engineering, cross-validation, keep rule).
- `titanic.py`: v1, the 11 features.
- `titanic_v2.py`: v2, adds the group-fate feature, built inside each fold from training labels only.
- `error_analysis.py`, `tree_analysis.py`: analysis scripts; graphs go to `plots/`.
- `submissions.csv`, `submissions_v2.csv`: Kaggle submissions of v1 (0.77751) and v2 (0.78708).
