import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import common

PLOTS_DIR = 'plots'

# Reference palette (light mode)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
RED = '#e34948'


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(width=9, height=5):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("Saved", path)


def out_of_fold_predictions(X, y, columns):
    # Each row is predicted by a model that did not train on it.
    folds = StratifiedKFold(n_splits=common.N_FOLDS, shuffle=True, random_state=common.RANDOM_SEED)
    probability = cross_val_predict(common.build_model(), X[columns], y['Survived'], cv=folds,
                                    method='predict_proba')[:, 1]
    result = X.copy()
    result['Survived'] = y['Survived']
    result['probability'] = probability
    result['predicted'] = (probability > 0.5).astype(int)
    result['correct'] = result['predicted'] == result['Survived']
    result['group'] = np.select([result['Sex_encoded'] == 1, result['is_male_adult'] == 1],
                                ['female', 'adult man'], 'boy')
    return result


def plot_errors_by_group_and_class(result):
    counts = (result.groupby(['group', 'Pclass'])['correct']
              .agg(right='sum', total='count')
              .reindex(pd.MultiIndex.from_product([['female', 'boy', 'adult man'], [1, 2, 3]])))
    counts['wrong'] = counts['total'] - counts['right']
    labels = [f"{group}\nclass {pclass}" for group, pclass in counts.index]
    x = np.arange(len(counts))
    fig, ax = new_figure(10, 5)
    ax.bar(x, counts['right'], width=0.6, color=BLUE, label='predicted right', edgecolor=SURFACE, linewidth=2)
    ax.bar(x, counts['wrong'], width=0.6, bottom=counts['right'], color=ORANGE, label='predicted wrong',
           edgecolor=SURFACE, linewidth=2)
    for i, (wrong, total) in enumerate(zip(counts['wrong'], counts['total'])):
        ax.text(i, total + 4, f"{int(wrong)} wrong of {int(total)}", ha='center', color=INK_SECONDARY, fontsize=8)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel('passengers', color=INK_SECONDARY)
    ax.set_title('Where the model is wrong: passengers by group and class', loc='left', color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc='upper left')
    save(fig, 'errors_by_group_and_class.png')
    return counts


def plot_probability_distribution(result):
    fig, ax = new_figure(9, 5)
    bins = np.linspace(0, 1, 21)
    outcomes = [result.loc[result['Survived'] == survived, 'probability'] for survived in [0, 1]]
    ax.hist(outcomes, bins=bins, color=[ORANGE, BLUE], label=['died', 'survived'], rwidth=0.85,
            edgecolor=SURFACE, linewidth=1)
    ax.axvline(0.5, color=INK_SECONDARY, linewidth=1, linestyle='--')
    ax.text(0.505, ax.get_ylim()[1] * 0.95, 'threshold 0.5', color=INK_SECONDARY, fontsize=9, va='top')
    ax.set_xlabel('predicted probability of survival', color=INK_SECONDARY)
    ax.set_ylabel('passengers', color=INK_SECONDARY)
    ax.set_title('Predicted probability, split by what actually happened', loc='left', color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    save(fig, 'probability_by_outcome.png')


def plot_weights(X, y, columns):
    model = common.build_model().fit(X[columns], y['Survived'])
    weights = pd.Series(model.coef_[0], index=columns).sort_values()
    fig, ax = new_figure(9, 5.5)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    colors = [BLUE if w > 0 else RED for w in weights]
    ax.barh(weights.index, weights.values, color=colors, height=0.6)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_xlim(weights.min() - 0.5, weights.max() + 0.5)
    for i, w in enumerate(weights.values):
        ax.text(w + (0.05 if w >= 0 else -0.05), i, f"{w:+.2f}", va='center',
                ha='left' if w >= 0 else 'right', color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel('weight (log-odds; positive raises survival)', color=INK_SECONDARY)
    ax.set_title(f'Model weights (bias b = {model.intercept_[0]:+.2f})', loc='left', color=INK, fontsize=12)
    save(fig, 'weights.png')
    return weights, model.intercept_[0]


def accuracy_ceiling(result, columns):
    # Passengers with identical features get the same prediction, so each feature combination
    # can at best be right for its majority outcome.
    combos = result.groupby(columns)['Survived'].agg(['sum', 'count'])
    best_right = np.maximum(combos['sum'], combos['count'] - combos['sum']).sum()
    return best_right / len(result), len(combos)


def describe_type(row):
    group = 'female' if row['Sex_encoded'] == 1 else ('adult man' if row['is_male_adult'] == 1 else 'boy')
    pclass = next(k for k in [1, 2, 3] if row[f'Pclass_{k}'] == 1)
    port = next(p for p in ['C', 'Q', 'S'] if row[f'Embarked_{p}'] == 1)
    family = next(name for col, name in [('is_single', 'single'), ('is_of_small_family', 'small fam'),
                                         ('is_of_large_family', 'large fam')] if row[col] == 1)
    return f"{group}, {pclass}, {port}, {family}"


def plot_passenger_types(result, columns, top=15):
    types = result.groupby(columns)['Survived'].agg(survived='sum', total='count').reset_index()
    types['died'] = types['total'] - types['survived']
    types['majority'] = np.maximum(types['survived'], types['died'])
    types['minority'] = types['total'] - types['majority']
    types['label'] = types.apply(describe_type, axis=1)
    shown = types.sort_values('total', ascending=False).head(top).iloc[::-1]
    fig, ax = new_figure(10, 6.5)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    y_pos = np.arange(len(shown))
    ax.barh(y_pos, shown['majority'], height=0.6, color=BLUE, label='majority outcome (can be predicted right)',
            edgecolor=SURFACE, linewidth=2)
    ax.barh(y_pos, shown['minority'], left=shown['majority'], height=0.6, color=ORANGE,
            label='minority outcome (always wrong for every model)', edgecolor=SURFACE, linewidth=2)
    for i, (_, row) in enumerate(shown.iterrows()):
        majority = 'survived' if row['survived'] > row['died'] else 'died'
        ax.text(row['total'] + 2, i, f"{int(row['survived'])} survived, {int(row['died'])} died -> predict {majority}",
                va='center', color=INK_SECONDARY, fontsize=8)
    ax.set_yticks(y_pos, shown['label'], fontsize=8)
    ax.set_xlim(0, shown['total'].max() * 1.6)
    ax.set_xlabel('passengers of this type (group, class, port, family)', color=INK_SECONDARY)
    ax.set_title(f'The {top} largest of {len(types)} passenger types', loc='left', color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc='lower right', fontsize=9)
    save(fig, 'passenger_types.png')
    return types


def export_cell(result, group, pclass, path):
    # All passengers of one cell, with the untouched train.csv columns (Age not filled) next to the prediction.
    cell = result[(result['group'] == group) & (result['Pclass'] == pclass)]
    raw = common.get_train_data_from_csv().drop(columns=['Survived'])
    added = cell[['PassengerId', 'correct', 'Survived', 'predicted', 'probability', 'Title', 'family_count',
                  'ticket_group_size']]
    export = added.merge(raw, on='PassengerId').sort_values(['correct', 'Survived', 'probability'])
    export['correct'] = export['correct'].astype(int)
    export['probability'] = export['probability'].round(3)
    export.to_csv(path, index=False)
    print(f"Saved {path}: {len(export)} rows, {export['correct'].sum()} right and "
          f"{(export['correct'] == 0).sum()} wrong")
    return export


def error_analysis(X_train, y_train):
    columns = common.fitting_columns(X_train)
    result = out_of_fold_predictions(X_train, y_train, columns)
    print(f"Out-of-fold accuracy: {result['correct'].mean():.3f} ({(~result['correct']).sum()} wrong of {len(result)})")
    ceiling, combos = accuracy_ceiling(result, columns)
    print(f"Feature combinations: {combos}; best possible accuracy with these features: {ceiling:.3f}")
    wrong = result[~result['correct']]
    print(f"Wrong predictions: {(wrong['predicted'] == 0).sum()} survivors predicted to die, "
          f"{(wrong['predicted'] == 1).sum()} who died predicted to survive")
    plot_errors_by_group_and_class(result)
    plot_probability_distribution(result)
    plot_passenger_types(result, columns)
    weights, bias = plot_weights(X_train, y_train, columns)
    print("Weights:", weights.round(2).to_dict(), "bias:", round(bias, 2))
    export_cell(result, 'female', 3, 'female_3rd_class.csv')
    return result


if __name__ == '__main__':
    X, y = common.retrieve_X_And_y()
    common.feature_engineering(X)
    X_train, X_test, y_train, y_test = common.split_train_test(X, y)
    error_analysis(X_train, y_train)
