# Gradient boosting on the Ridge model's columns. Written by Claude; the Ridge model is in common.py.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import common
from data_analysis import BLUE, GRID, INK, INK_SECONDARY, ORANGE, SURFACE, save, style_axes

# The Ridge model's category columns, passed to the trees as categories (no one-hot, no target encoding)
CATEGORY_COLUMNS = list(common.ONE_HOT_REFERENCES) + common.TARGET_ENCODED_COLUMNS


def build_boosting(learning_rate=0.1, max_iter=100):
    # Same columns as the Ridge model (option A). Numbers and 0/1 flags as they are: a split only uses the order.
    # Categories become codes 0, 1, 2, ... and are declared categorical; a category unseen in training gets -1,
    # which the trees treat as missing.
    numeric = common.NUMERIC_COLUMNS + common.PASSTHROUGH_COLUMNS
    preprocess = ColumnTransformer([
        ('numbers', 'passthrough', numeric),
        ('categories', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORY_COLUMNS),
    ])
    is_category = [False] * len(numeric) + [True] * len(CATEGORY_COLUMNS)
    model = HistGradientBoostingRegressor(learning_rate=learning_rate, max_iter=max_iter, early_stopping=False,
                                          categorical_features=is_category, random_state=common.RANDOM_SEED)
    return Pipeline([('preprocess', preprocess), ('model', model)])


def tree_count_curves(learning_rates=(0.03, 0.1, 0.3), max_trees=1000):
    # Training and validation RMSE after each tree, averaged over the same 50 folds as everywhere else.
    X, y = common.retrieve_X_And_y()
    X_train, y, _ = common.feature_engineering(X, y, common.get_kaggle_test_data_from_csv())
    folds = list(common.repeated_folds().split(X_train))
    curves = {}
    for rate in learning_rates:
        train_rmse = np.zeros(max_trees)
        validation_rmse = np.zeros(max_trees)
        for train_index, validation_index in folds:
            pipeline = build_boosting(rate, max_trees).fit(X_train.iloc[train_index], y.iloc[train_index])
            for part, index, total in [('train', train_index, train_rmse),
                                       ('validation', validation_index, validation_rmse)]:
                features = pipeline.named_steps['preprocess'].transform(X_train.iloc[index])
                for step, prediction in enumerate(pipeline.named_steps['model'].staged_predict(features)):
                    total[step] += np.sqrt(np.mean((prediction - y.iloc[index].values) ** 2))
        curves[rate] = (train_rmse / len(folds), validation_rmse / len(folds))
        best = int(np.argmin(curves[rate][1]))
        print(f"learning rate {rate}: lowest validation RMSE {curves[rate][1][best]:.4f} at {best + 1} trees; "
              f"at 100 trees {curves[rate][1][99]:.4f}; training RMSE at 1000 trees {curves[rate][0][-1]:.4f}")
    return curves


def plot_tree_count_curves(curves, ridge_rmse=0.1118):
    fig, axes = plt.subplots(1, len(curves), figsize=(15, 5), dpi=120, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (rate, (train_rmse, validation_rmse)) in zip(axes, curves.items()):
        style_axes(ax)
        trees = np.arange(1, len(train_rmse) + 1)
        ax.plot(trees, validation_rmse, color=BLUE, linewidth=2, label='validation')
        ax.plot(trees, train_rmse, color=ORANGE, linewidth=2, linestyle='--', label='training')
        ax.axhline(ridge_rmse, color=INK_SECONDARY, linewidth=1, linestyle=':')
        ax.text(1.1, ridge_rmse, f"Ridge {ridge_rmse:.4f}", ha='left', va='bottom', color=INK_SECONDARY,
                fontsize=8)
        ax.axvline(100, color=GRID, linewidth=1)
        best = int(np.argmin(validation_rmse))
        ax.plot(best + 1, validation_rmse[best], 'o', color=BLUE, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2)
        ax.annotate(f"lowest {validation_rmse[best]:.4f}\nat {best + 1} trees", (best + 1, validation_rmse[best]),
                    xytext=(0, -28), textcoords='offset points', ha='center', color=INK, fontsize=8)
        ax.text(len(trees), validation_rmse[-1] + 0.004, 'validation', ha='right', va='bottom',
                color=INK_SECONDARY, fontsize=8)
        ax.text(len(trees), train_rmse[-1], 'training', ha='right', va='bottom', color=INK_SECONDARY, fontsize=8)
        ax.set_xscale('log')
        ax.set_ylim(0, 0.3)
        ax.set_xlabel('number of trees (log scale; grey line: 100, the default)', color=INK_SECONDARY)
        ax.set_title(f"learning rate {rate}", loc='left', color=INK, fontsize=11)
    axes[0].set_ylabel('RMSE (log price), mean of 50 folds', color=INK_SECONDARY)
    axes[0].legend(loc='upper right', frameon=False, labelcolor=INK_SECONDARY)
    fig.suptitle('Gradient boosting on the Ridge model\'s columns: error against the number of trees',
                 x=0.01, ha='left', color=INK, fontsize=12)
    save(fig, 'boosting_tree_count.png')


if __name__ == '__main__':
    plot_tree_count_curves(tree_count_curves())
