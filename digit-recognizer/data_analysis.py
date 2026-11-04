# Graphs and checks for the digit models. Written by Claude; the model code is the user's (digit_recognizer.py).

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import digit_recognizer as dr

PLOTS_DIR = 'plots'

# Reference palette (light mode), same as the other problems
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BLUE = '#2a78d6'


def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("Saved", path)


def baseline():
    X, y = dr.get_X_y(dr.get_data_from_csv())
    return dr.get_model_and_test_data(dr.divide_by_255(X), y)


def confused_pairs(model, X_val, y_val, top=8):
    # Mistakes per (true, predicted) pair, most common first, and per true digit
    predicted = model.predict(X_val)
    matrix = confusion_matrix(y_val, predicted, labels=range(10))
    pairs = [(matrix[t, p], t, p) for t in range(10) for p in range(10) if t != p]
    pairs.sort(reverse=True)
    print(f"Mistakes: {(predicted != y_val).sum()} of {len(y_val)}")
    print("Most common mistakes (true -> predicted):")
    for count, t, p in pairs[:top]:
        print(f"  {t} -> {p}: {count}")
    per_digit = pd.DataFrame({'images': matrix.sum(axis=1), 'correct': np.diag(matrix)}, index=range(10))
    per_digit['accuracy'] = per_digit['correct'] / per_digit['images']
    print(per_digit.round(3).sort_values('accuracy').to_string())
    return matrix


def plot_mistakes(matrix, file_name='baseline_mistakes.png', title='Logistic regression: mistakes on the held-out images'):
    # Off-diagonal counts only (the correct ones would swamp the colour scale), one light-to-dark hue
    mistakes = matrix.astype(float).copy()
    np.fill_diagonal(mistakes, np.nan)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(mistakes, cmap='Blues', vmin=0)
    for t in range(10):
        for p in range(10):
            if t != p and mistakes[t, p] > 0:
                dark = mistakes[t, p] > np.nanmax(mistakes) * 0.6
                ax.text(p, t, int(mistakes[t, p]), ha='center', va='center', fontsize=8,
                        color=SURFACE if dark else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel('predicted digit', color=INK_SECONDARY)
    ax.set_ylabel('true digit', color=INK_SECONDARY)
    ax.set_title(title, loc='left', color=INK, fontsize=12)
    fig.colorbar(image, ax=ax, shrink=0.8).outline.set_visible(False)
    save(fig, file_name)


if __name__ == '__main__':
    model, X_val, y_val = baseline()
    plot_mistakes(confused_pairs(model, X_val, y_val))
