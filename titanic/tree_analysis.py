import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.tree import DecisionTreeClassifier

import error_analysis as ea
import common

MIN_SAMPLES_LEAF_VALUES = [2, 5, 10, 20, 40, 80]
MISSING_AGE_SENTINEL = -1
# Ways to give the tree Age; each is added on top of the 11 logistic regression features.
AGE_VARIANTS = {
    'C: fill + flag': ['Age', 'age_missing'],
    'E: sentinel -1': ['age_sentinel'],
    'F: NaN, tree decides': ['age_with_nan'],
}
FOREST_AGE_COLUMNS = AGE_VARIANTS['C: fill + flag']


def build_tree(**settings):
    return DecisionTreeClassifier(random_state=common.RANDOM_SEED, **settings)


def build_forest(**settings):
    # Defaults: 100 trees, bootstrap rows, sqrt(features) candidates per split.
    return RandomForestClassifier(random_state=common.RANDOM_SEED, n_jobs=-1, **settings)


def add_age_variants(X, age_missing):
    # Only for the tree. Age itself keeps the title-mean fill that is_male_adult relies on.
    X['age_missing'] = age_missing.astype(int)
    X['age_sentinel'] = X['Age'].where(~age_missing, MISSING_AGE_SENTINEL)
    X['age_with_nan'] = X['Age'].where(~age_missing)


def min_samples_leaf_scores(X, y, columns, values=MIN_SAMPLES_LEAF_VALUES, model=None, per_fold=False):
    # Same 50 folds for every value and every column set, so all points are directly comparable.
    model = build_tree() if model is None else model
    train_scores, validation_scores = validation_curve(model, X[columns], y['Survived'],
                                                       param_name='min_samples_leaf', param_range=values,
                                                       cv=common.repeated_folds(), scoring='accuracy')
    if per_fold:
        return train_scores, validation_scores
    return train_scores.mean(axis=1), validation_scores.mean(axis=1)


def compare_folds(name, new, base):
    # Paired comparison on the same folds, judged by the keep rule.
    gain = new.mean() - base.mean()
    win_share = (new > base).mean()
    keep = gain >= common.MIN_ACCURACY_GAIN and win_share >= common.MIN_FOLD_WIN_SHARE
    print(f"{name}: gain {gain:+.3f}, better in {win_share:.0%} of folds, keep rule passed: {keep}")


def plot_curve(ax, values, train, validation, logistic, reference=None):
    ea.style_axes(ax)
    x = np.arange(len(values))
    if reference is not None:
        ax.plot(x, reference, color=ea.BASELINE, linewidth=1.5, label='validation, 11 features (no Age)')
    # Training labels above the line, validation labels below, so they never collide.
    for scores, color, label, offset, va in [(train, ea.ORANGE, 'training', 0.003, 'bottom'),
                                             (validation, ea.BLUE, 'validation', -0.003, 'top')]:
        ax.plot(x, scores, color=color, linewidth=2, marker='o', label=f'{label} accuracy')
        for i, score in enumerate(scores):
            ax.text(i, score + offset, f"{score:.3f}", ha='center', va=va, color=color, fontsize=7)
    ax.axhline(logistic, color=ea.INK_SECONDARY, linewidth=1, linestyle='--')
    ax.text(-0.3, logistic + 0.002, f'logistic regression {logistic:.3f}', ha='left', va='bottom',
            color=ea.INK_SECONDARY, fontsize=7)
    ax.set_xlim(-0.4, len(values) - 0.6)
    ax.set_xticks(x, values)
    ax.set_xlabel('min_samples_leaf', color=ea.INK_SECONDARY)


def plot_age_variant_curves(X, y, base_columns, values=MIN_SAMPLES_LEAF_VALUES):
    logistic = cross_val_score(common.build_model(), X[base_columns], y['Survived'],
                               cv=common.repeated_folds(), scoring='accuracy').mean()
    _, reference = min_samples_leaf_scores(X, y, base_columns, values)
    results = {name: min_samples_leaf_scores(X, y, base_columns + columns, values)
               for name, columns in AGE_VARIANTS.items()}

    fig, axes = plt.subplots(1, len(results), figsize=(15, 5.5), dpi=120, sharey=True)
    fig.patch.set_facecolor(ea.SURFACE)
    for ax, (name, (train, validation)) in zip(axes, results.items()):
        plot_curve(ax, values, train, validation, logistic, reference)
        ax.set_title(name, loc='left', color=ea.INK, fontsize=11)
    all_scores = np.concatenate([np.concatenate(scores) for scores in results.values()] + [reference])
    axes[0].set_ylim(all_scores.min() - 0.01, all_scores.max() + 0.01)
    axes[0].set_ylabel('accuracy (mean of 50 folds)', color=ea.INK_SECONDARY)
    axes[0].legend(frameon=False, labelcolor=ea.INK_SECONDARY, loc='upper right', fontsize=8)
    fig.suptitle('Decision tree with Age added: three ways to handle missing ages', x=0.01, ha='left',
                 color=ea.INK, fontsize=13)
    ea.save(fig, 'tree_curve_age_variants.png')

    print(f"Logistic regression validation: {logistic:.3f}")
    print("11 features, no Age:  " + ", ".join(f"{v}: {s:.3f}" for v, s in zip(values, reference)))
    for name, (train, validation) in results.items():
        print(f"{name}")
        for value, t, v in zip(values, train, validation):
            print(f"  min_samples_leaf {value:>3}: training {t:.3f}, validation {v:.3f}")
    return results


def plot_forest_curves(X, y, base_columns, values=MIN_SAMPLES_LEAF_VALUES):
    logistic = cross_val_score(common.build_model(), X[base_columns], y['Survived'],
                               cv=common.repeated_folds(), scoring='accuracy')
    column_sets = {'11 features': base_columns, '11 features + Age (C: fill + flag)': base_columns + FOREST_AGE_COLUMNS}
    results = {}
    for name, columns in column_sets.items():
        _, tree_validation = min_samples_leaf_scores(X, y, columns, values)
        train, validation = min_samples_leaf_scores(X, y, columns, values, model=build_forest(), per_fold=True)
        results[name] = (train, validation, tree_validation)

    fig, axes = plt.subplots(1, len(results), figsize=(13, 5.5), dpi=120, sharey=True)
    fig.patch.set_facecolor(ea.SURFACE)
    for ax, (name, (train, validation, tree_validation)) in zip(axes, results.items()):
        plot_curve(ax, values, train.mean(axis=1), validation.mean(axis=1), logistic.mean(), tree_validation)
        ax.set_title(name, loc='left', color=ea.INK, fontsize=11)
    axes[0].lines[0].set_label('validation, single tree (same columns)')
    all_scores = np.concatenate([np.concatenate([r[0].mean(axis=1), r[1].mean(axis=1), r[2]]) for r in results.values()])
    axes[0].set_ylim(all_scores.min() - 0.01, all_scores.max() + 0.01)
    axes[0].set_ylabel('accuracy (mean of 50 folds)', color=ea.INK_SECONDARY)
    axes[0].legend(frameon=False, labelcolor=ea.INK_SECONDARY, loc='upper right', fontsize=8)
    fig.suptitle('Random forest (100 trees) validation curves', x=0.01, ha='left', color=ea.INK, fontsize=13)
    ea.save(fig, 'forest_curves.png')

    print(f"Logistic regression validation: {logistic.mean():.3f}")
    best = {}
    for name, (train, validation, tree_validation) in results.items():
        print(name)
        for value, t, v, tv in zip(values, train.mean(axis=1), validation.mean(axis=1), tree_validation):
            print(f"  min_samples_leaf {value:>3}: forest training {t:.3f}, forest validation {v:.3f}, "
                  f"single tree validation {tv:.3f}")
        best_idx = int(validation.mean(axis=1).argmax())
        best[name] = validation[best_idx]
        compare_folds(f"  forest at min_samples_leaf {values[best_idx]} vs logistic regression",
                      validation[best_idx], logistic)
    without_age, with_age = best.values()
    compare_folds("Forest best with Age vs forest best without Age", with_age, without_age)
    return results


if __name__ == '__main__':
    X, y = common.retrieve_X_And_y()
    age_missing = X['Age'].isna()
    common.feature_engineering(X)
    base_columns = common.fitting_columns(X)
    add_age_variants(X, age_missing)
    X_train, X_test, y_train, y_test = common.split_train_test(X, y)
    #plot_age_variant_curves(X_train, y_train, base_columns)
    plot_forest_curves(X_train, y_train, base_columns)
