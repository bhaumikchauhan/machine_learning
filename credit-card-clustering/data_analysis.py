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


def cluster_profiles(k, version=1):
    # Size and median of the key columns per cluster, in the original units (dollars, frequencies, months).
    import credit_card_clustering as ccc
    data = load()
    pipeline = ccc.build_pipeline(version)
    pipeline.set_params(model__n_clusters=k)
    pipeline.fit(data.drop(columns=['CUST_ID']))
    labels = pipeline.named_steps['model'].labels_
    profile = data[PROFILE_COLUMNS].groupby(labels).median()
    profile.insert(0, 'share', pd.Series(labels).value_counts(normalize=True))
    profile = profile.sort_values('share', ascending=False)
    print(f"k = {k}")
    print(profile.round(2).to_string())
    return profile


# Six categorical colours in the reference palette's fixed order
CLUSTER_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']


def plot_pca_clusters(file_name='pca_clusters.png'):
    # The user's PCA coordinates (first 2 axes of the 17 scaled columns), one dot per customer, coloured by cluster.
    import credit_card_clustering as ccc
    coordinates, labels = ccc.pca_coordinates()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    for cluster in sorted(set(labels)):
        points = coordinates[labels == cluster]
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.5, color=CLUSTER_COLOURS[cluster % len(CLUSTER_COLOURS)],
                   linewidths=0, label=f"cluster {cluster} ({len(points) / len(labels):.0%})")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, markerscale=2.5, loc='best')
    ax.set_xlabel('PCA axis 1', color=INK_SECONDARY)
    ax.set_ylabel('PCA axis 2', color=INK_SECONDARY)
    ax.set_title(f'Customers on the first 2 PCA axes, {len(set(labels))} clusters (version 1)', loc='left',
                 color=INK, fontsize=12)
    save(fig, file_name)


def compare_versions():
    # Version 1 (MINIMUM_PAYMENTS blank = 0) against version 2 (KNNImputer). Cluster numbers are arbitrary, so each
    # version 2 cluster is first matched to the version 1 cluster it shares most customers with.
    from scipy.optimize import linear_sum_assignment
    import credit_card_clustering as ccc
    data = load()
    X = data.drop(columns=['CUST_ID'])
    labels = {}
    for version in (1, 2):
        pipeline = ccc.build_pipeline(version).fit(X)
        labels[version] = pipeline.named_steps['model'].labels_
        if version == 2:
            filled = pipeline.named_steps['preprocess'].named_transformers_['account_size']
            log_scaled = filled[:-1].transform(X[ccc.ACCOUNT_SIZE_COLUMNS])
            scaler = filled.named_steps['scale']
            dollars = np.expm1(scaler.inverse_transform(filled.named_steps['fill'].transform(log_scaled)))[:, 0]
    overlap = pd.crosstab(labels[1], labels[2])
    rows, cols = linear_sum_assignment(-overlap.values)
    match = {overlap.columns[c]: overlap.index[r] for r, c in zip(rows, cols)}
    matched = pd.Series(labels[2]).map(match).values
    changed = matched != labels[1]
    blank = X['MINIMUM_PAYMENTS'].isna().values
    print(f"Customers who change cluster: {changed.sum()} of {len(X)} ({changed.mean():.1%})")
    print(f"  of the {blank.sum()} with a blank MINIMUM_PAYMENTS: {changed[blank].sum()}")
    print(f"  of the other {(~blank).sum()}: {changed[~blank].sum()}")
    print(f"Version 2 fills for the blanks: median ${np.median(dollars[blank]):,.0f}, "
          f"quartiles ${np.percentile(dollars[blank], 25):,.0f} to ${np.percentile(dollars[blank], 75):,.0f}")
    print("Cluster shares, version 1 against version 2 (matched numbers):")
    print(pd.DataFrame({'version 1': pd.Series(labels[1]).value_counts(normalize=True),
                        'version 2': pd.Series(matched).value_counts(normalize=True)}).round(3).to_string())
    return changed


SEGMENT_NAMES = {0: 'only cash borrowers', 1: 'moderate users', 2: 'high usage buyers', 3: 'cash and credit users',
                 4: 'frequent buyers, low balance and limit', 5: 'inactive'}  # k-means numbers, n_init=10


def match_labels(reference, other):
    # Renumber `other` so each of its clusters takes the number of the reference cluster it shares most customers with
    from scipy.optimize import linear_sum_assignment
    overlap = pd.crosstab(reference, other)
    rows, cols = linear_sum_assignment(-overlap.values)
    mapping = {overlap.columns[c]: overlap.index[r] for r, c in zip(rows, cols)}
    return pd.Series(other).map(mapping).values


def compare_ward_with_kmeans():
    # k-means (n_init=10) against Ward on the same 17 scaled columns: who changes cluster, and the Ward profiles
    import credit_card_clustering as ccc
    data = load()
    X = data.drop(columns=['CUST_ID'])
    kmeans = ccc.build_pipeline().fit(X).named_steps['model'].labels_
    ward_pipeline = ccc.build_ward_pipeline().fit(X)
    ward = match_labels(kmeans, ward_pipeline.named_steps['model'].labels_)
    changed = ward != kmeans
    print(f"Customers who change cluster: {changed.sum()} of {len(X)} ({changed.mean():.1%})")
    moves = pd.crosstab(pd.Series(kmeans, name='k-means').map(SEGMENT_NAMES),
                        pd.Series(ward, name='Ward').map(SEGMENT_NAMES))
    print(moves.to_string())
    profile = data[PROFILE_COLUMNS].groupby(ward).median()
    profile.insert(0, 'share', pd.Series(ward).value_counts(normalize=True))
    profile.insert(0, 'k-means share', pd.Series(kmeans).value_counts(normalize=True))
    profile.index = profile.index.map(SEGMENT_NAMES)
    print(profile.round(2).to_string())
    return ward_pipeline


def plot_dendrogram(ward_pipeline, file_name='ward_dendrogram.png'):
    # The last 30 merges of Ward's tree, rebuilt with scipy on the same scaled columns (same merges as scikit-learn)
    from scipy.cluster.hierarchy import dendrogram, linkage
    X = load().drop(columns=['CUST_ID'])
    tree = linkage(ward_pipeline.named_steps['preprocess'].transform(X), method='ward')
    cut = (tree[-6, 2] + tree[-5, 2]) / 2  # between the merge that leaves 6 clusters and the one that leaves 5
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    dendrogram(tree, truncate_mode='lastp', p=30, ax=ax, color_threshold=0, above_threshold_color=BLUE,
               leaf_font_size=8, show_leaf_counts=True)
    ax.axhline(cut, color=MUTED, linewidth=1, linestyle='--')
    ax.annotate('cut for 6 clusters', (ax.get_xlim()[1], cut), xytext=(-4, 4), textcoords='offset points',
                ha='right', color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel('customers in each branch (last 30 merges shown)', color=INK_SECONDARY)
    ax.set_ylabel('merge distance (Ward)', color=INK_SECONDARY)
    ax.set_title('Ward dendrogram, version 1 columns', loc='left', color=INK, fontsize=12)
    save(fig, file_name)


if __name__ == '__main__':
    # raw_against_log()  # done: plots/purchases_raw_against_log.png
    # plot_inertia(inertia_by_k())  # done: plots/inertia_by_k.png
    # for k in (6, 7, 8):  # done: profiles in feature_notes.md
    #     cluster_profiles(k)
    # plot_pca_clusters()  # done: plots/pca_clusters.png
    # compare_versions()  # done: version 1 kept
    plot_dendrogram(compare_ward_with_kmeans())
