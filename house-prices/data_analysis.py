# Graphs used to decide on features. Written by Claude; the model code is in house_prices.py.

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict

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


# Short names for the MSSubClass codes, from data_description.txt.
MSSUBCLASS_NAMES = {
    20: '1-story new', 30: '1-story old', 40: '1-story attic', 45: '1.5-story unfin.',
    50: '1.5-story fin.', 60: '2-story new', 70: '2-story old', 75: '2.5-story',
    80: 'split/multi-level', 85: 'split foyer', 90: 'duplex', 120: '1-story PUD',
    150: '1.5-story PUD', 160: '2-story PUD', 180: 'PUD multilevel', 190: '2-family conv.',
}


# Where a point's label goes, relative to the point: offset in points, then horizontal and vertical alignment.
LABEL_PLACES = {
    'above': ((0, 9), 'center', 'bottom'),
    'below': ((0, -9), 'center', 'top'),
    'left': ((-8, 0), 'right', 'center'),
    'right': ((8, 0), 'left', 'center'),
}


def plot_mean_by_mssubclass(df, column, label, unit_format, label_places, file_name):
    # Each code drawn at its numeric value: what the model sees if the code is used as a number.
    stats = df.groupby('MSSubClass')[column].agg(['mean', 'size'])
    fig, ax = new_figure(11, 5.5)
    ax.scatter(stats.index, stats['mean'], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    # label_places: codes whose label would collide with a neighbour's label or dot when placed above
    for code, mean, n in zip(stats.index, stats['mean'], stats['size']):
        offset, ha, va = LABEL_PLACES[label_places.get(code, 'above')]
        ax.annotate(f"{code} {MSSUBCLASS_NAMES[code]}\n{unit_format(mean)}, n={n}", (code, mean),
                    xytext=offset, textcoords='offset points', ha=ha, va=va,
                    color=INK_SECONDARY, fontsize=7.5)
    ax.set_xlim(5, 205)
    ax.set_ylim(0, stats['mean'].max() * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: unit_format(v))
    ax.set_xlabel('MSSubClass code (as a number)', color=INK_SECONDARY)
    ax.set_ylabel(label, color=INK_SECONDARY)
    ax.set_title(f"{label.capitalize()} per MSSubClass code, drawn at the code's numeric value (train.csv)",
                 loc='left', color=INK, fontsize=11)
    save(fig, file_name)


# All zones in data_description.txt, in its order. The data writes C as 'C (all)'. 'missing' is an empty value.
MSZONING_NAMES = {
    'A': 'Agriculture', 'C (all)': 'Commercial', 'FV': 'Floating Village Res.', 'I': 'Industrial',
    'RH': 'Residential High Density', 'RL': 'Residential Low Density', 'RP': 'Residential Low Density Park',
    'RM': 'Residential Medium Density', 'missing': 'missing',
}


def plot_mszoning_counts(train, test):
    zones = list(MSZONING_NAMES)
    counts = {name: df['MSZoning'].fillna('missing').value_counts().reindex(zones, fill_value=0)
              for name, df in [('train.csv', train), ('test.csv', test)]}
    fig, ax = new_figure(11, 5.5)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    y = range(len(zones))
    bar_height = 0.38
    for i, ((name, count), color) in enumerate(zip(counts.items(), [BLUE, ORANGE])):
        positions = [p + (i - 0.5) * (bar_height + 0.02) for p in y]
        ax.barh(positions, count, height=bar_height, color=color, label=name)
        for position, n in zip(positions, count):
            ax.text(n + 8, position, str(n), va='center', color=INK_SECONDARY, fontsize=8)
    ax.set_yticks(list(y), [f"{zone}  {MSZONING_NAMES[zone]}" if zone != 'missing' else 'missing'
                            for zone in zones])
    ax.invert_yaxis()
    ax.set_xlim(0, max(c.max() for c in counts.values()) * 1.1)
    ax.set_xlabel('number of houses', color=INK_SECONDARY)
    ax.legend(frameon=False, loc='lower right', labelcolor=INK_SECONDARY)
    ax.set_title('Number of houses per MSZoning value', loc='left', color=INK, fontsize=11)
    save(fig, 'mszoning_counts.png')


ZONE_COLORS = {'RL': BLUE, 'RM': ORANGE, 'C (all)': '#1baf7a', 'FV': '#8a5cc4', 'RH': '#c79a1b'}
LOCATION_COLUMNS = ['Neighborhood', 'Condition1', 'Condition2']


def plot_missing_mszoning(train, test):
    # Uses train and test together: MSZoning is an input, not the label, so test rows may help fill it.
    houses = pd.concat([train, test])
    known = houses[houses['MSZoning'].notna()]
    missing = houses[houses['MSZoning'].isna()]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), dpi=120, width_ratios=[1, 1.3])
    fig.patch.set_facecolor(SURFACE)

    # Left: zone mix of the known houses with the same neighborhood and conditions as each missing house.
    style_axes(left)
    left.grid(axis='y', visible=False)
    left.grid(axis='x', color=GRID, linewidth=0.8)
    groups = missing.groupby(LOCATION_COLUMNS)['Id'].apply(list)
    for row, (location, ids) in enumerate(groups.items()):
        same = known
        for column, value in zip(LOCATION_COLUMNS, location):
            same = same[same[column] == value]
        counts = same['MSZoning'].value_counts()
        start = 0
        for zone, n in counts.items():
            left.barh(row, n, left=start, height=0.5, color=ZONE_COLORS[zone], edgecolor=SURFACE, linewidth=2)
            if n >= 8:
                left.text(start + n / 2, row, f"{zone}\n{n}", ha='center', va='center', color=SURFACE, fontsize=8)
            start += n
        narrow = counts.iloc[-1] < 8  # last segment too narrow for its label inside: put it after the bar
        note = (f"{counts.index[-1]} {counts.iloc[-1]}   " if narrow else '') + \
            f"missing: Id {', '.join(map(str, ids))}"
        left.text(start + 1.5, row, note, va='center', color=INK_SECONDARY, fontsize=8)
    left.set_yticks(range(len(groups)), [f"{n}\n{c1} / {c2}" for n, c1, c2 in groups.index])
    left.invert_yaxis()
    left.set_xlim(0, 115)
    left.set_xlabel('known houses with the same Neighborhood / Condition1 / Condition2', color=INK_SECONDARY)
    left.set_title('Zones of the matching houses', loc='left', color=INK, fontsize=11)

    # Right: lot area of every IDOTRR house by zone, with the three missing ones.
    style_axes(right)
    idotrr = houses[houses['Neighborhood'] == 'IDOTRR']
    zones = ['RM', 'C (all)', 'missing']
    for x, zone in enumerate(zones):
        rows = idotrr[idotrr['MSZoning'].isna()] if zone == 'missing' else idotrr[idotrr['MSZoning'] == zone]
        jitter = [((i * 37) % 21 - 10) / 40 for i in range(len(rows))]  # fixed spread, no random draw
        right.scatter([x + j for j in jitter], rows['LotArea'], s=40, color=ZONE_COLORS.get(zone, INK),
                      edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if zone == 'missing':
            for j, (house_id, area) in zip(jitter, zip(rows['Id'], rows['LotArea'])):
                right.annotate(f"Id {house_id}", (x + j, area), xytext=(8, 0), textcoords='offset points',
                               va='center', color=INK_SECONDARY, fontsize=8)
    right.set_xticks(range(len(zones)), [f"{z}\n(n={len(idotrr[idotrr['MSZoning'] == z]) if z != 'missing' else 3})"
                                         for z in zones])
    right.set_xlim(-0.6, 2.8)
    right.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    right.set_ylabel('LotArea (sq ft)', color=INK_SECONDARY)
    right.set_title('IDOTRR: lot area by zone (train + test)', loc='left', color=INK, fontsize=11)
    save(fig, 'mszoning_missing.png')


def plot_big_lots(train, test, min_lot_area=30000):
    houses = pd.concat([train, test])
    big = houses[houses['LotArea'] > min_lot_area].copy()
    big['zone'] = big['MSZoning'].fillna('missing')
    counts = pd.crosstab(big['Neighborhood'], big['zone'])
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    colors = dict(ZONE_COLORS, missing=INK)
    zones = [z for z in ['RL', 'RM', 'missing'] if z in counts.columns]
    fig, ax = new_figure(10, 5.5)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    start = pd.Series(0, index=counts.index)
    for zone in zones:
        ax.barh(counts.index, counts[zone], left=start, height=0.6, color=colors[zone], label=zone,
                edgecolor=SURFACE, linewidth=2)
        start += counts[zone]
    for neighborhood, total in start.items():
        missing_ids = big[(big['Neighborhood'] == neighborhood) & (big['zone'] == 'missing')]['Id']
        note = f"{total}" + (f"   missing: Id {', '.join(map(str, missing_ids))}" if len(missing_ids) else '')
        ax.text(total + 0.1, neighborhood, note, va='center', color=INK_SECONDARY, fontsize=8)
    ax.set_xlim(0, start.max() + 3)
    ax.set_xlabel('number of houses (train + test)', color=INK_SECONDARY)
    ax.legend(frameon=False, loc='lower right', labelcolor=INK_SECONDARY, title='MSZoning',
              title_fontproperties={'size': 9})
    ax.set_title(f"Houses with LotArea above {min_lot_area:,} sq ft, by neighborhood and zone ({len(big)} houses)",
                 loc='left', color=INK, fontsize=11)
    save(fig, 'big_lots_by_neighborhood.png')


def plot_zones_by_neighborhood(train, test):
    houses = pd.concat([train, test])
    houses['zone'] = houses['MSZoning'].fillna('missing')
    counts = pd.crosstab(houses['Neighborhood'], houses['zone'])
    zones = ['RL', 'RM', 'C (all)', 'FV', 'RH', 'missing']
    counts = counts[zones]
    # Sorted by the share of RL, lowest first at the top.
    counts = counts.loc[(counts['RL'] / counts.sum(axis=1)).sort_values().index]
    colors = dict(ZONE_COLORS, missing=INK)
    fig, ax = new_figure(11, 8)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    start = pd.Series(0, index=counts.index)
    for zone in zones:
        ax.barh(counts.index, counts[zone], left=start, height=0.65, color=colors[zone], label=zone,
                edgecolor=SURFACE, linewidth=1.5)
        start += counts[zone]
    for neighborhood, row in counts.iterrows():
        parts = [f"{zone} {n}" for zone, n in row.items() if n > 0]
        ax.text(start[neighborhood] + 3, neighborhood, ', '.join(parts), va='center', color=INK_SECONDARY,
                fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, start.max() * 1.25)
    ax.set_xlabel('number of houses (train + test)', color=INK_SECONDARY)
    ax.legend(frameon=False, loc='lower right', labelcolor=INK_SECONDARY, title='MSZoning',
              title_fontproperties={'size': 9})
    ax.set_title('Zones per neighborhood, sorted by share of RL (lowest at the top)', loc='left', color=INK,
                 fontsize=11)
    save(fig, 'zones_by_neighborhood.png')


LOT_CONFIG_NAMES = {'Inside': 'Inside lot', 'Corner': 'Corner lot', 'CulDSac': 'Cul-de-sac',
                    'FR2': 'Frontage on 2 sides', 'FR3': 'Frontage on 3 sides'}


def plot_lot_frontage(train, test):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), dpi=120, width_ratios=[1.2, 1])
    fig.patch.set_facecolor(SURFACE)

    # Left: frontage against sale price (train only: test has no price).
    style_axes(left)
    left.grid(axis='x', color=GRID, linewidth=0.8)
    known = train[train['LotFrontage'].notna()]
    left.scatter(known['LotFrontage'], known['SalePrice'], s=14, color=BLUE, alpha=0.5, linewidth=0)
    left.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
    left.set_xlabel('LotFrontage (feet of street along the lot)', color=INK_SECONDARY)
    left.set_ylabel('SalePrice', color=INK_SECONDARY)
    corr = known['LotFrontage'].corr(known['SalePrice'])
    left.set_title(f"Frontage vs price, train.csv ({len(known)} houses with a value, correlation {corr:.2f})",
                   loc='left', color=INK, fontsize=10)

    # Right: share of houses with no LotFrontage, per lot configuration.
    style_axes(right)
    right.grid(axis='y', visible=False)
    right.grid(axis='x', color=GRID, linewidth=0.8)
    houses = pd.concat([train, test])
    stats = houses.groupby('LotConfig')['LotFrontage'].agg(missing=lambda s: s.isna().sum(), n='size')
    stats['share'] = stats['missing'] / stats['n']
    stats = stats.sort_values('share')
    names = [LOT_CONFIG_NAMES[c] for c in stats.index]
    right.barh(names, stats['share'] * 100, height=0.6, color=ORANGE)
    for name, share, missing, n in zip(names, stats['share'], stats['missing'], stats['n']):
        right.text(share * 100 + 1, name, f"{share:.0%}  ({missing} of {n})", va='center', color=INK_SECONDARY,
                   fontsize=8)
    overall = houses['LotFrontage'].isna().mean() * 100
    right.axvline(overall, color=MUTED, linewidth=1, linestyle='--')
    right.text(overall + 0.5, len(stats) - 0.45, f"all houses {overall:.0f}%", color=MUTED, fontsize=8,
               va='bottom')
    right.set_ylim(-0.5, len(stats) - 0.1)
    right.set_xlim(0, 75)
    right.set_xlabel('houses with no LotFrontage value (%)', color=INK_SECONDARY)
    right.set_title('Missing LotFrontage by LotConfig (train + test)', loc='left', color=INK, fontsize=10)
    save(fig, 'lot_frontage.png')


OUTLIER_IDS = [524, 1299]  # the two largest houses, both sold far below the trend


def plot_living_area_outliers(train):
    fig, ax = new_figure(11, 5.5)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    outliers = train[train['Id'].isin(OUTLIER_IDS)]
    rest = train[~train['Id'].isin(OUTLIER_IDS)]
    ax.scatter(rest['GrLivArea'], rest['SalePrice'], s=14, color=BLUE, alpha=0.45, linewidth=0)
    ax.scatter(outliers['GrLivArea'], outliers['SalePrice'], s=60, color=ORANGE, edgecolor=SURFACE, linewidth=2,
               zorder=3)
    for house_id, area, price in zip(outliers['Id'], outliers['GrLivArea'], outliers['SalePrice']):
        ax.annotate(f"Id {house_id}: {area:,} sq ft, ${price / 1000:.0f}k", (area, price), xytext=(-8, 10),
                    textcoords='offset points', ha='right', color=INK_SECONDARY, fontsize=8)
    x = np.array([train['GrLivArea'].min(), train['GrLivArea'].max()])
    lines = [('all 1,460 houses', train, INK, '-'), (f"without Ids {', '.join(map(str, OUTLIER_IDS))}", rest,
                                                       ORANGE, '--')]
    for label, rows, color, style in lines:
        slope, intercept = np.polyfit(rows['GrLivArea'], rows['SalePrice'], 1)
        ax.plot(x, intercept + slope * x, color=color, linewidth=2, linestyle=style,
                label=f"straight-line fit, {label}: ${slope:.0f} per sq ft")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel('GrLivArea (above-ground living area, sq ft)', color=INK_SECONDARY)
    ax.set_ylabel('SalePrice', color=INK_SECONDARY)
    ax.legend(frameon=False, loc='upper left', labelcolor=INK_SECONDARY, fontsize=9)
    ax.set_title('Living area vs price, train.csv: what the two largest houses do to a straight-line fit',
                 loc='left', color=INK, fontsize=11)
    save(fig, 'living_area_outliers.png')


def plot_lot_area(train):
    # Without the two removed outliers.
    houses = train[~train['Id'].isin(OUTLIER_IDS)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=120, height_ratios=[1, 1.6])
    fig.patch.set_facecolor(SURFACE)
    panels = [('LotArea as recorded', houses['LotArea'], 'LotArea (sq ft)'),
              ('log(LotArea)', np.log(houses['LotArea']), 'log(LotArea)')]
    for column, (title, values, label) in enumerate(panels):
        top, bottom = axes[0, column], axes[1, column]
        style_axes(top)
        top.hist(values, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
        top.set_title(f"{title}: number of houses (skew {values.skew():.1f})", loc='left', color=INK, fontsize=10)
        top.set_ylabel('houses', color=INK_SECONDARY)
        style_axes(bottom)
        bottom.grid(axis='x', color=GRID, linewidth=0.8)
        bottom.scatter(values, houses['SalePrice'], s=12, color=BLUE, alpha=0.45, linewidth=0)
        slope, intercept = np.polyfit(values, houses['SalePrice'], 1)
        x = np.array([values.min(), values.max()])
        bottom.plot(x, intercept + slope * x, color=INK, linewidth=2)
        bottom.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
        bottom.set_xlabel(label, color=INK_SECONDARY)
        bottom.set_ylabel('SalePrice', color=INK_SECONDARY)
        bottom.set_title(f"{title} vs price, with a straight-line fit (correlation {values.corr(houses['SalePrice']):.2f})",
                         loc='left', color=INK, fontsize=10)
    for ax in axes[:, 0]:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    fig.suptitle(f"LotArea, train.csv without Ids {', '.join(map(str, OUTLIER_IDS))} ({len(houses)} houses)",
                 x=0.01, ha='left', color=INK, fontsize=12)
    save(fig, 'lot_area.png')


def plot_price_by_category(train, panels, file_name, title):
    # panels: (column, {value: label}) pairs; a None value stands for an empty cell, which may mean "none".
    houses = train[~train['Id'].isin(OUTLIER_IDS)]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(9, 6 * len(panels)), 5), dpi=120,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (column, labels) in zip(np.atleast_1d(axes), panels):
        style_axes(ax)
        for x, (value, label) in enumerate(labels.items()):
            rows = houses[houses[column].isna()] if value is None else houses[houses[column] == value]
            prices = rows['SalePrice']
            jitter = [((i * 37) % 41 - 20) / 70 for i in range(len(prices))]  # fixed spread, no random draw
            ax.scatter([x + j for j in jitter], prices, s=10, color=BLUE, alpha=0.35, linewidth=0)
            ax.plot([x - 0.35, x + 0.35], [prices.median()] * 2, color=INK, linewidth=2)
            ax.text(x, -45000, f"median ${prices.median() / 1000:.0f}k\nn={len(prices)}", ha='center', va='top',
                    color=INK_SECONDARY, fontsize=8)
        ax.set_xticks(range(len(labels)), list(labels.values()))
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(-110000, houses['SalePrice'].max() * 1.05)
        ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k" if v >= 0 else '')
        ax.set_title(column, loc='left', color=INK, fontsize=11)
    np.atleast_1d(axes)[0].set_ylabel('SalePrice (black line: median)', color=INK_SECONDARY)
    fig.suptitle(title, x=0.01, ha='left', color=INK, fontsize=12)
    save(fig, file_name)


def plot_fr3_houses(train, test):
    houses = pd.concat([train[~train['Id'].isin(OUTLIER_IDS)].assign(source='train'), test.assign(source='test')])
    fig, ax = new_figure(11, 6)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    others = houses[~houses['LotConfig'].isin(['FR2', 'FR3'])]
    ax.scatter(others['YearBuilt'], others['LotArea'], s=8, color=BASELINE, alpha=0.5, linewidth=0,
               label='all other lots')
    fr2 = houses[houses['LotConfig'] == 'FR2']
    ax.scatter(fr2['YearBuilt'], fr2['LotArea'], s=22, color=BLUE, alpha=0.7, linewidth=0, label='FR2 (train + test)')
    for source, marker in [('train', 'o'), ('test', 'D')]:
        fr3 = houses[(houses['LotConfig'] == 'FR3') & (houses['source'] == source)]
        ax.scatter(fr3['YearBuilt'], fr3['LotArea'], s=60, marker=marker, color=ORANGE, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=f"FR3, {source}.csv ({len(fr3)})")
        for _, row in fr3.iterrows():
            price = f", ${row['SalePrice'] / 1000:.0f}k" if source == 'train' else ''
            left = row['Id'] == 1360  # same year and lot size as test Id 1666
            ax.annotate(f"{row['Id']} {row['BldgType']}{price}", (row['YearBuilt'], row['LotArea']),
                        xytext=(-6, 10) if left else (6, 4), textcoords='offset points',
                        ha='right' if left else 'left', color=INK_SECONDARY, fontsize=7.5)
    ax.set_ylim(0, 25000)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    ax.set_xlabel('YearBuilt', color=INK_SECONDARY)
    ax.set_ylabel('LotArea (sq ft; lots over 25k not shown)', color=INK_SECONDARY)
    ax.legend(frameon=False, loc='upper left', labelcolor=INK_SECONDARY, fontsize=9)
    ax.set_title('FR3 houses (frontage on 3 sides): year built and lot size, with building type and train price',
                 loc='left', color=INK, fontsize=11)
    save(fig, 'fr3_houses.png')


def plot_neighborhoods(train, test):
    houses = train[~train['Id'].isin(OUTLIER_IDS)]
    order = houses.groupby('Neighborhood')['SalePrice'].median().sort_values().index
    test_counts = test['Neighborhood'].value_counts()
    fig, ax = new_figure(11, 10)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    for y, name in enumerate(order):
        prices = houses.loc[houses['Neighborhood'] == name, 'SalePrice']
        jitter = [((i * 37) % 41 - 20) / 70 for i in range(len(prices))]  # fixed spread, no random draw
        ax.scatter(prices, [y + j for j in jitter], s=8, color=BLUE, alpha=0.35, linewidth=0)
        ax.plot([prices.median()] * 2, [y - 0.38, y + 0.38], color=INK, linewidth=2)
        ax.text(760000, y, f"n={len(prices)} / {test_counts.get(name, 0)}", va='center', color=INK_SECONDARY,
                fontsize=8)
    ax.axvline(houses['SalePrice'].median(), color=ORANGE, linewidth=1, linestyle='--')
    ax.text(houses['SalePrice'].median() + 5000, len(order) - 0.4, 'overall median', color=ORANGE, fontsize=8)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlim(0, 840000)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k")
    ax.set_xlabel('SalePrice (black line: median; right: houses in train / test)', color=INK_SECONDARY)
    ax.set_title('Sale price by Neighborhood, sorted by median, train.csv without Ids 524, 1299',
                 loc='left', color=INK, fontsize=11)
    save(fig, 'neighborhoods.png')


BLDG_TYPE_NAMES = {'1Fam': '1Fam\nsingle-family', '2fmCon': '2fmCon\n2-family conv.', 'Duplex': 'Duplex',
                   'Twnhs': 'Twnhs\ntownhouse inside', 'TwnhsE': 'TwnhsE\ntownhouse end'}


HOUSE_STYLE_NAMES = {'1Story': '1Story', '1.5Fin': '1.5Fin', '1.5Unf': '1.5Unf', '2Story': '2Story',
                     '2.5Fin': '2.5Fin', '2.5Unf': '2.5Unf', 'SFoyer': 'SFoyer\nsplit foyer',
                     'SLvl': 'SLvl\nsplit level'}


def plot_vs_mssubclass(train, test, column, names, file_name):
    houses = pd.concat([train, test])
    counts = pd.crosstab(houses['MSSubClass'], houses[column])[list(names)]
    fig, ax = new_figure(max(9, 1.5 * len(names)), 8)
    ax.grid(axis='y', visible=False)
    for x, bldg_type in enumerate(counts.columns):
        for y, code in enumerate(counts.index):
            n = counts.loc[code, bldg_type]
            if n == 0:
                continue
            ax.scatter(x, y, s=n * 2.5, color=BLUE, alpha=0.3, linewidth=0)
            ax.text(x, y, n, ha='center', va='center', color=INK, fontsize=8)
    ax.set_xticks(range(len(counts.columns)), [names[c] for c in counts.columns])
    ax.set_yticks(range(len(counts.index)), [f"{c} {MSSUBCLASS_NAMES[c]}" for c in counts.index])
    ax.set_xlim(-0.6, len(counts.columns) - 0.4)
    ax.set_ylim(len(counts.index) - 0.5, -0.5)
    ax.set_title(f"Houses by MSSubClass and {column}, train.csv + test.csv", loc='left', color=INK, fontsize=11)
    save(fig, file_name)


def plot_column_overview(train, test, skip):
    # Share of log-price spread explained by each column on its own: squared correlation for numbers
    # (missing filled with the median), group means for categories (missing as its own group).
    houses = train[~train['Id'].isin(OUTLIER_IDS)]
    log_price = np.log(houses['SalePrice'])
    rows = []
    for column in houses.columns.drop(skip):
        values = houses[column]
        if pd.api.types.is_numeric_dtype(values):
            share = values.fillna(values.median()).corr(log_price) ** 2
        else:
            group_mean = log_price.groupby(values.fillna('NA')).transform('mean')
            share = 1 - ((log_price - group_mean) ** 2).sum() / ((log_price - log_price.mean()) ** 2).sum()
        rows.append((column, share, pd.api.types.is_numeric_dtype(values),
                     train[column].isna().sum(), test[column].isna().sum()))
    rows.sort(key=lambda row: row[1])
    fig, ax = new_figure(11, 15)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    for y, (column, share, numeric, missing_train, missing_test) in enumerate(rows):
        ax.barh(y, share, height=0.7, color=BLUE if numeric else ORANGE)
        if missing_train or missing_test:
            ax.text(share + 0.005, y, f"missing {missing_train} / {missing_test}", va='center',
                    color=INK_SECONDARY, fontsize=7.5)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 0.8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(handles=[Patch(color=BLUE, label='number'), Patch(color=ORANGE, label='category')],
              loc='lower right', frameon=False, labelcolor=INK_SECONDARY)
    ax.set_xlabel('share of log-price spread explained by the column alone (missing: train / test)',
                  color=INK_SECONDARY)
    ax.set_title('Columns not analysed yet, train.csv without Ids 524, 1299', loc='left', color=INK,
                 fontsize=11)
    save(fig, 'column_overview.png')


ANALYSED_COLUMNS = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
                    'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                    'Condition2', 'BldgType', 'HouseStyle']


CONDITION_NAMES = {
    'Norm': 'Norm\nnothing nearby', 'Artery': 'Artery\narterial st.', 'Feedr': 'Feedr\nfeeder st.',
    'RRNn': 'RRNn\nnear N-S rail', 'RRAn': 'RRAn\nnext to N-S rail', 'RRNe': 'RRNe\nnear E-W rail',
    'RRAe': 'RRAe\nnext to E-W rail', 'PosN': 'PosN\nnear park etc.', 'PosA': 'PosA\nnext to park etc.',
}


def plot_conditions(train, test):
    # A house counts under every condition in Condition1 or Condition2, so the 30 houses with two appear twice.
    # Norm only counts houses with nothing nearby: Condition2 is Norm for almost every house.
    houses = train[~train['Id'].isin(OUTLIER_IDS)]
    fig, ax = new_figure(13, 5.5)
    for x, (value, label) in enumerate(CONDITION_NAMES.items()):
        second = value if value != 'Norm' else None
        prices = houses.loc[(houses['Condition1'] == value) | (houses['Condition2'] == second), 'SalePrice']
        in_test = ((test['Condition1'] == value) | (test['Condition2'] == second)).sum()
        jitter = [((i * 37) % 41 - 20) / 70 for i in range(len(prices))]  # fixed spread, no random draw
        ax.scatter([x + j for j in jitter], prices, s=10, color=BLUE, alpha=0.35, linewidth=0)
        ax.plot([x - 0.35, x + 0.35], [prices.median()] * 2, color=INK, linewidth=2)
        ax.text(x, -45000, f"median ${prices.median() / 1000:.0f}k\nn={len(prices)} / {in_test}", ha='center',
                va='top', color=INK_SECONDARY, fontsize=8)
    ax.set_xticks(range(len(CONDITION_NAMES)), list(CONDITION_NAMES.values()))
    ax.set_xlim(-0.6, len(CONDITION_NAMES) - 0.4)
    ax.set_ylim(-110000, houses['SalePrice'].max() * 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v / 1000:.0f}k" if v >= 0 else '')
    ax.set_ylabel('SalePrice (black line: median)', color=INK_SECONDARY)
    ax.set_title('Sale price by nearby condition (Condition1 or Condition2), train.csv without Ids 524, 1299; '
                 'n = houses in train / test', loc='left', color=INK, fontsize=11)
    save(fig, 'conditions.png')


# Raw columns already decided: used by the current model (directly or through a derived column), or dropped.
DECIDED_COLUMNS = ['Id', 'MSSubClass', 'Neighborhood', 'LotArea', 'OverallQual', 'OverallCond',
                   'YearRemodAdd', 'YrSold', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'ExterQual',
                   'GarageCars', 'TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'LotShape', 'Utilities',
                   'BldgType', 'HouseStyle', 'MSZoning', 'Functional', 'SaleCondition',
                   'CentralAir', 'Exterior1st', 'Condition1', 'Condition2', 'GarageType', 'GarageFinish',
                   'GarageArea', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'BsmtQual', 'BsmtCond',
                   'BsmtExposure', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                   'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu']


def plot_leftover_ranking(train, skip, top=20):
    # Rank columns by the share of the current model's leftover they explain: what each would add that the model
    # doesn't know yet. Leftover = log price - out-of-fold prediction. Numbers in 10 equal-count bands, categories
    # by value, missing as its own group; the share is adjusted for the number of groups.
    X, y = common.retrieve_X_And_y()
    X_train, y, _ = common.feature_engineering(X, y, common.get_kaggle_test_data_from_csv())
    prediction = cross_val_predict(common.build_pipeline(), X_train, y,
                                   cv=KFold(common.N_FOLDS, shuffle=True, random_state=common.RANDOM_SEED))
    leftover = pd.Series(y.values - prediction, index=X_train.index)
    houses = train.loc[X_train.index]
    n = len(leftover)
    total = ((leftover - leftover.mean()) ** 2).sum()
    rows = []
    for column in houses.columns.drop(skip):
        values = houses[column]
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > 12:
            bands = pd.qcut(values.rank(method='first'), 10, labels=False)
            groups = pd.Series(bands, index=values.index).astype(float).fillna(-1).astype(str)
        else:
            # astype(object) first: in pandas 3, astype(str) keeps missing values missing
            groups = values.astype(object).where(values.notna(), 'missing').astype(str)
        k = groups.nunique()
        explained = 1 - ((leftover - leftover.groupby(groups).transform('mean')) ** 2).sum() / total
        rows.append((column, 1 - (1 - explained) * (n - 1) / (n - k), values.isna().sum()))
    rows = sorted(rows, key=lambda row: row[1], reverse=True)[:top][::-1]
    fig, ax = new_figure(10, 7)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    for y_pos, (column, share, missing) in enumerate(rows):
        ax.barh(y_pos, share, height=0.7, color=BLUE)
        ax.text(share + 0.001, y_pos, f"{share:.1%}" + (f"  (missing {missing})" if missing else ''), va='center',
                color=INK_SECONDARY, fontsize=8)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of the current model's leftover explained (adjusted for the number of groups)",
                  color=INK_SECONDARY)
    ax.set_title(f"Unused columns: what they would add to the current model (top {top}), train.csv without "
                 "Ids 524, 1299", loc='left', color=INK, fontsize=11)
    save(fig, 'leftover_ranking.png')
    for column, share, missing in rows[::-1]:
        print(f"{column:15s} {share:6.1%}")


if __name__ == '__main__':
    train = common.get_train_data_from_csv()
    test = common.get_kaggle_test_data_from_csv()
    plot_leftover_ranking(train, DECIDED_COLUMNS + ['SalePrice'])
    plot_price_by_category(train, [('Street', {'Grvl': 'Grvl\ngravel', 'Pave': 'Pave\npaved'}),
                                   ('Alley', {None: 'empty\n(no alley access)', 'Grvl': 'Grvl\ngravel',
                                              'Pave': 'Pave\npaved'})],
                           'street_alley.png', 'Sale price by road access (Street) and alley access (Alley), '
                           'train.csv without Ids 524, 1299')
    plot_price_by_category(train, [('LotShape', {'Reg': 'Reg\nregular', 'IR1': 'IR1\nslightly irregular',
                                                 'IR2': 'IR2\nmoderately irregular', 'IR3': 'IR3\nirregular'})],
                           'lot_shape.png', 'Sale price by LotShape, train.csv without Ids 524, 1299')
    plot_price_by_category(train, [('LandContour', {'Lvl': 'Lvl\nlevel', 'Bnk': 'Bnk\nbanked',
                                                    'HLS': 'HLS\nhillside', 'Low': 'Low\ndepression'})],
                           'land_contour.png', 'Sale price by LandContour, train.csv without Ids 524, 1299')
    plot_price_by_category(train, [('LotConfig', {'Inside': 'Inside\ninside lot', 'Corner': 'Corner\ncorner lot',
                                                  'CulDSac': 'CulDSac\ncul-de-sac', 'FR2': 'FR2\nfrontage 2 sides',
                                                  'FR3': 'FR3\nfrontage 3 sides'})],
                           'lot_config.png', 'Sale price by LotConfig, train.csv without Ids 524, 1299')
    plot_price_by_category(train, [('LandSlope', {'Gtl': 'Gtl\ngentle', 'Mod': 'Mod\nmoderate',
                                                  'Sev': 'Sev\nsevere'})],
                           'land_slope.png', 'Sale price by LandSlope, train.csv without Ids 524, 1299')
    plot_column_overview(train, test, ANALYSED_COLUMNS + ['SalePrice'])
    plot_price_by_category(train, [('BldgType', BLDG_TYPE_NAMES)], 'bldg_type.png',
                           'Sale price by BldgType, train.csv without Ids 524, 1299')
    plot_vs_mssubclass(train, test, 'BldgType', BLDG_TYPE_NAMES, 'bldg_type_vs_mssubclass.png')
    plot_price_by_category(train, [('HouseStyle', HOUSE_STYLE_NAMES)], 'house_style.png',
                           'Sale price by HouseStyle, train.csv without Ids 524, 1299')
    plot_vs_mssubclass(train, test, 'HouseStyle', HOUSE_STYLE_NAMES, 'house_style_vs_mssubclass.png')
    plot_conditions(train, test)
    plot_neighborhoods(train, test)
    plot_fr3_houses(train, test)
    plot_lot_area(train)
    plot_living_area_outliers(train)
    plot_mszoning_counts(train, test)
    plot_missing_mszoning(train, test)
    plot_big_lots(train, test)
    plot_zones_by_neighborhood(train, test)
    plot_lot_frontage(train, test)
    plot_mean_by_mssubclass(train, 'GrLivArea', 'mean above-ground living area (sq ft)',
                            lambda v: f"{v:.0f} sq ft", {30: 'below', 80: 'below', 85: 'below'},
                            'mssubclass_living_area.png')
    plot_mean_by_mssubclass(train, 'SalePrice', 'mean sale price',
                            lambda v: f"${v / 1000:.0f}k", {30: 'below', 45: 'below', 50: 'below', 80: 'right',
                                                             85: 'left', 90: 'right', 180: 'below'},
                            'mssubclass_sale_price.png')
