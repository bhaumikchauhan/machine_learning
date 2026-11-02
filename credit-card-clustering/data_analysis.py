# Graphs and checks used to decide on scaling and columns. Written by Claude; the model code is the user's.

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = 'plots'
DATA_PATH = os.path.join('kaggle', 'input', 'ccdata', 'CC GENERAL.csv')

# Reference palette (light mode), same as house-prices
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
BLUE = '#2a78d6'


def load():
    return pd.read_csv(DATA_PATH)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("Saved", path)


def standard_scores(values):
    # What StandardScaler does: distance from the mean in standard deviations
    return (values - values.mean()) / values.std(ddof=0)


def raw_against_log(column='PURCHASES'):
    # Histogram of the column as it is and after log(1 + x), with the biggest customer's distance in sd for each.
    raw = load()[column]
    logged = np.log1p(raw)
    zeros = (raw == 0).sum()
    print(f"{column}: {zeros} customers at 0 ({zeros / len(raw):.0%})")
    for name, values in [('raw', raw), ('log(1 + x)', logged)]:
        z = standard_scores(values)
        print(f"  {name}: biggest customer at {z.max():.1f} sd, smallest at {z.min():.1f} sd, "
              f"customers beyond 3 sd: {(z.abs() > 3).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    for ax, values, title in [(axes[0], raw, f'{column} ($)'), (axes[1], logged, f'log(1 + {column})')]:
        style_axes(ax)
        ax.hist(values, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=1)
        z_max = standard_scores(values).max()
        ax.annotate(f"biggest customer: {z_max:.1f} sd from the mean", (values.max(), 0), xytext=(-4, 130),
                    textcoords='offset points', ha='right', color=INK, fontsize=9,
                    arrowprops=dict(arrowstyle='-', color=MUTED))
        ax.set_xlabel(title, color=INK_SECONDARY)
        ax.set_title(title, loc='left', color=INK, fontsize=12)
    axes[0].set_ylabel('customers', color=INK_SECONDARY)
    axes[1].annotate(f"{zeros} customers at 0", (0, (logged == 0).sum()), xytext=(10, -4),
                     textcoords='offset points', color=INK, fontsize=9)
    save(fig, f'{column.lower()}_raw_against_log.png')


def inertia_by_k(k_values=range(2, 16)):
    # Uses the user's Pipeline; only n_clusters changes. Returns inertia and the smallest cluster's share per k.
    import credit_card_clustering as ccc
    X = load().drop(columns=['CUST_ID'])
    rows = []
    for k in k_values:
        pipeline = ccc.build_pipeline()
        pipeline.set_params(model__n_clusters=k)
        pipeline.fit(X)
        model = pipeline.named_steps['model']
        shares = pd.Series(model.labels_).value_counts(normalize=True)
        rows.append({'k': k, 'inertia': model.inertia_, 'smallest_share': shares.min(), 'largest_share': shares.max()})
    table = pd.DataFrame(rows).set_index('k')
    table['drop'] = -table['inertia'].diff()
    table['drop_share'] = table['drop'] / table['inertia'].shift()
    print(table.round({'inertia': 0, 'drop': 0, 'smallest_share': 3, 'largest_share': 3, 'drop_share': 3}).to_string())
    return table


def plot_inertia(table, allowed=(5, 10), file_name='inertia_by_k.png'):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.axvspan(allowed[0] - 0.5, allowed[1] + 0.5, color=GRID, alpha=0.5, linewidth=0)
    ax.annotate(f"allowed range ({allowed[0]} to {allowed[1]})", (allowed[0] - 0.4, table['inertia'].max()),
                color=INK_SECONDARY, fontsize=9, va='top')
    ax.plot(table.index, table['inertia'], color=BLUE, linewidth=2, marker='o', markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xticks(table.index)
    ax.set_xlabel('number of clusters (k)', color=INK_SECONDARY)
    ax.set_ylabel('inertia (sum of squared distances to the centre)', color=INK_SECONDARY)
    ax.set_title('Inertia by number of clusters (version 1)', loc='left', color=INK, fontsize=12)
    save(fig, file_name)


PROFILE_COLUMNS = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                   'PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT', 'CREDIT_LIMIT', 'TENURE']


def cluster_profiles(k):
    # Size and median of the key columns per cluster, in the original units (dollars, frequencies, months).
    import credit_card_clustering as ccc
    data = load()
    pipeline = ccc.build_pipeline()
    pipeline.set_params(model__n_clusters=k)
    pipeline.fit(data.drop(columns=['CUST_ID']))
    labels = pipeline.named_steps['model'].labels_
    profile = data[PROFILE_COLUMNS].groupby(labels).median()
    profile.insert(0, 'share', pd.Series(labels).value_counts(normalize=True))
    profile = profile.sort_values('share', ascending=False)
    print(f"k = {k}")
    print(profile.round(2).to_string())
    return profile


if __name__ == '__main__':
    # raw_against_log()  # done: plots/purchases_raw_against_log.png
    # plot_inertia(inertia_by_k())  # done: plots/inertia_by_k.png
    for k in (6, 7, 8):
        cluster_profiles(k)
