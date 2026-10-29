# Gradient boosting on the Ridge model's columns. Written by Claude; the Ridge model is in common.py.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
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


def raw_data():
    # Option B: the CSV columns as they are, same rows (outliers removed) and order as the Ridge model,
    # so the 50 folds hold the same houses.
    X, y = common.retrieve_X_And_y()
    X, y = common.ids_remove_outlier(X, common.target_log_transform(y))
    X = X.drop(columns=['Id'])
    X['MSSubClass'] = X['MSSubClass'].astype(str)
    return X, y


def build_boosting_raw(X, learning_rate=0.1, max_iter=100):
    # Text columns as categories; missing values stay missing (OrdinalEncoder keeps NaN, the trees handle it)
    categories = list(X.select_dtypes(exclude='number').columns)
    numeric = list(X.select_dtypes(include='number').columns)
    preprocess = ColumnTransformer([
        ('numbers', 'passthrough', numeric),
        ('categories', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categories),
    ])
    is_category = [False] * len(numeric) + [True] * len(categories)
    model = HistGradientBoostingRegressor(learning_rate=learning_rate, max_iter=max_iter, early_stopping=False,
                                          categorical_features=is_category, random_state=common.RANDOM_SEED)
    return Pipeline([('preprocess', preprocess), ('model', model)])


def engineered_data():
    # Option A: the Ridge model's columns
    X, y = common.retrieve_X_And_y()
    X_train, y, _ = common.feature_engineering(X, y, common.get_kaggle_test_data_from_csv())
    return X_train, y


def tree_count_curves(learning_rates=(0.03, 0.1, 0.3), max_trees=1000, option='A'):
    # Training and validation RMSE after each tree, averaged over the same 50 folds as everywhere else.
    X_train, y = engineered_data() if option == 'A' else raw_data()
    folds = list(common.repeated_folds().split(X_train))
    curves = {}
    for rate in learning_rates:
        train_rmse = np.zeros(max_trees)
        validation_rmse = np.zeros(max_trees)
        for train_index, validation_index in folds:
            pipeline = build_boosting(rate, max_trees) if option == 'A' else build_boosting_raw(X_train, rate, max_trees)
            pipeline.fit(X_train.iloc[train_index], y.iloc[train_index])
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


def compare_with_ridge(learning_rate, trees):
    # Option B at one setting against the Ridge model (the version in place), fold by fold
    X_raw, y = raw_data()
    scores = cross_validate(build_boosting_raw(X_raw, learning_rate, trees), X_raw, y, cv=common.repeated_folds(),
                            scoring='neg_root_mean_squared_error')
    boosting_rmse = -scores['test_score']
    X_train, y_train = engineered_data()
    ridge_rmse = -common.cross_validate_model(X_train, y_train)['test_neg_root_mean_squared_error']
    print(f"Ridge {ridge_rmse.mean():.4f}, boosting on raw columns {boosting_rmse.mean():.4f}")
    common.keep_rule(ridge_rmse, boosting_rmse)


BLEND_WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)


def blend_with_ridge(learning_rate=0.03, trees=329):
    # Ridge (current model) and boosting (raw columns) fitted on the same training houses in each fold;
    # the blend predicts (1 - w) * Ridge + w * boosting, in log price. Returns fold RMSEs per weight w.
    X_raw, y = raw_data()
    X_train, y_train = engineered_data()
    assert X_raw.index.equals(X_train.index)
    rmse = {w: [] for w in BLEND_WEIGHTS}
    for train_index, validation_index in common.repeated_folds().split(X_train):
        ridge = common.build_pipeline().fit(X_train.iloc[train_index], y_train.iloc[train_index])
        boosting = build_boosting_raw(X_raw, learning_rate, trees).fit(X_raw.iloc[train_index], y.iloc[train_index])
        ridge_prediction = ridge.predict(X_train.iloc[validation_index])
        boosting_prediction = boosting.predict(X_raw.iloc[validation_index])
        actual = y.iloc[validation_index].values
        for w in BLEND_WEIGHTS:
            prediction = (1 - w) * ridge_prediction + w * boosting_prediction
            rmse[w].append(np.sqrt(np.mean((prediction - actual) ** 2)))
    rmse = {w: np.array(scores) for w, scores in rmse.items()}
    for w, scores in rmse.items():
        print(f"boosting weight {w}: RMSE {scores.mean():.4f}, wins against Ridge {(scores < rmse[0.0]).mean():.0%}")
    print("50/50 average against Ridge alone:")
    common.keep_rule(rmse[0.0], rmse[0.5])
    return rmse


def plot_blend(rmse):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    weights = list(rmse)
    means = [rmse[w].mean() for w in weights]
    ax.plot(weights, means, color=BLUE, linewidth=2, marker='o', markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2)
    for w in (0.0, 0.5, 1.0):
        ax.annotate(f"{rmse[w].mean():.4f}", (w, rmse[w].mean()), xytext=(0, 10), textcoords='offset points',
                    ha='center', color=INK, fontsize=9)
    best = weights[int(np.argmin(means))]
    ax.annotate(f"lowest {rmse[best].mean():.4f}\nat weight {best}", (best, rmse[best].mean()), xytext=(0, 14),
                textcoords='offset points', ha='center', color=INK, fontsize=9)
    ax.set_ylim(min(means) - 0.002, max(means) + 0.002)
    ax.set_xticks(weights)
    ax.set_xlabel('weight on boosting (0 = Ridge alone, 1 = boosting alone)', color=INK_SECONDARY)
    ax.set_ylabel('RMSE (log price), mean of 50 folds', color=INK_SECONDARY)
    ax.set_title('Blending Ridge and boosting (raw columns)', loc='left', color=INK, fontsize=12)
    save(fig, 'blend_ridge_boosting.png')


def plot_tree_count_curves(curves, ridge_rmse=0.1118, title_columns="the Ridge model's columns",
                           file_name='boosting_tree_count.png'):
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
    fig.suptitle(f'Gradient boosting on {title_columns}: error against the number of trees',
                 x=0.01, ha='left', color=INK, fontsize=12)
    save(fig, file_name)


if __name__ == '__main__':
    # Already run:
    # option A: plot_tree_count_curves(tree_count_curves())
    # option B: plot_tree_count_curves(tree_count_curves(learning_rates=(0.03, 0.1), option='B'),
    #                                  title_columns='the raw columns', file_name='boosting_tree_count_raw.png')
    # option B against Ridge: compare_with_ridge(0.03, 329)
    plot_blend(blend_with_ridge())
