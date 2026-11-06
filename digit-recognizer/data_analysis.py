# Graphs and checks for the digit models. Written by Claude; the model code is the user's (digit_recognizer.py).

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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


def compare_epoch_choices(seeds=range(5)):
    # The dense network two ways, on the same seeds: 10 epochs on all 33,600 training images (last epoch kept)
    # against early stopping (best of 30 on the 6,720 early-stopping images). Judged by average held-out accuracy.
    X, y = dr.get_X_y(dr.get_data_from_csv())
    X = dr.divide_by_255(X)
    rows = []
    for seed in seeds:
        _, _, ten_epochs = dr.train_network(dr.build_dense_network, X, y, epochs=10, seed=seed,
                                            early_stopping=False, verbose=False)
        _, history, stopped = dr.train_network(dr.build_dense_network, X, y, epochs=30, seed=seed, verbose=False)
        best_epoch = max(history, key=lambda h: h['stop_accuracy'])['epoch']
        rows.append({'seed': seed, '10 epochs': ten_epochs, 'early stopping': stopped, 'best epoch': best_epoch})
        print(f"seed {seed}: 10 epochs {ten_epochs:.4f}, early stopping {stopped:.4f} (best epoch {best_epoch})")
    table = pd.DataFrame(rows).set_index('seed')
    print(table.to_string(float_format='{:.4f}'.format))
    for column in ['10 epochs', 'early stopping']:
        print(f"{column}: mean {table[column].mean():.4f}, sd {table[column].std():.4f}")
    gain = (table['early stopping'].mean() - table['10 epochs'].mean()) * 100
    print(f"Gain of early stopping: {gain:+.2f} points (keep rule: at least +0.50)")
    return table


def compare_networks(models, seeds=range(5), **training):
    # Each model on the same seeds, judged by average held-out accuracy. A model is a build function, or
    # (build function, its own training settings) when models differ in training, such as augmentation;
    # `training` applies to all of them (train_network's defaults otherwise: best of 30 epochs on the
    # early-stopping set). For the keep rule's noise check, each model after the first also gets its
    # disagreements with the first, per seed: images exactly one of the two gets right (~69 between two CNN seeds).
    X, y = dr.get_X_y(dr.get_data_from_csv())
    X = dr.divide_by_255(X)
    _, X_test, _, y_test = dr.split_data(X, y)
    truth = y_test.values
    table = pd.DataFrame(index=pd.Index(list(seeds), name='seed'))
    first_name, first_right = None, {}
    for name, model in models.items():
        build, settings = model if isinstance(model, tuple) else (model, {})
        for seed in seeds:
            network, _, accuracy = dr.train_network(build, X, y, seed=seed, verbose=False, **{**training, **settings})
            right = dr.predict_digits(network, X_test) == truth
            table.loc[seed, name] = accuracy
            line = f"{name}, seed {seed}: {accuracy:.4f}"
            if first_name is None or first_name == name:
                first_name, first_right[seed] = name, right
            else:
                table.loc[seed, f"{name} disagreements"] = disagreements = (right != first_right[seed]).sum()
                line += f", {disagreements} disagreements with {first_name}"
            print(line)
    for column in table.columns[table.columns.str.endswith('disagreements')]:
        table[column] = table[column].astype(int)
    print(table.to_string(float_format='{:.4f}'.format))
    for name in models:
        gain = f", gain over {first_name} {(table[name].mean() - table[first_name].mean()) * 100:+.2f} points" \
            if name != first_name else ""
        print(f"{name}: mean {table[name].mean():.4f}, sd {table[name].std():.4f}{gain}")
    return table


def plot_augmentation(n_images=6, n_versions=5):
    # A few training images (left column) and random versions of each from random_shift_rotate_zoom,
    # to check by eye that every version is still clearly the same digit
    X, y = dr.get_X_y(dr.get_data_from_csv())
    X = dr.divide_by_255(X).iloc[:n_images]
    torch.manual_seed(0)
    originals = torch.tensor(X.values, dtype=torch.float32)
    versions = [dr.random_shift_rotate_zoom(originals) for _ in range(n_versions)]
    fig, axes = plt.subplots(n_images, n_versions + 1, figsize=(1.3 * (n_versions + 1), 1.3 * n_images), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    for row in range(n_images):
        for column, images in enumerate([originals] + versions):
            ax = axes[row, column]
            ax.imshow(images[row].reshape(28, 28), cmap='Greys', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(GRID)
        axes[row, 0].set_ylabel(f"label {y.iloc[row]}", color=INK_SECONDARY, fontsize=8)
    axes[0, 0].set_title('original', color=INK_SECONDARY, fontsize=9)
    for column in range(1, n_versions + 1):
        axes[0, column].set_title(f'version {column}', color=MUTED, fontsize=9)
    fig.suptitle('Random shift (±3 px), rotation (±10°) and zoom (±10%), a new draw per image', color=INK,
                 fontsize=11, x=0.02, ha='left')
    save(fig, 'augmentation_examples.png')


def vote(probabilities):
    # probabilities: networks × images × 10. Each network votes for its most probable digit, the most votes wins
    # (a tie goes to the lower digit). Returns the digits and the number of ties.
    votes = probabilities.argmax(axis=2)
    counts = np.stack([(votes == digit).sum(axis=0) for digit in range(10)])  # votes per digit, 10 × images
    ties = ((counts == counts.max(axis=0)).sum(axis=0) > 1).sum()
    return counts.argmax(axis=0), ties


def compare_combinations(seeds=range(5)):
    # Votes against averaged probabilities for combining several CNNs, measured on the early-stopping set
    # (the same 6,720 images for every seed), so the held-out set stays for the final judgement.
    X, y = dr.get_X_y(dr.get_data_from_csv())
    X = dr.divide_by_255(X)
    X_train, _, y_train, _ = dr.split_data(X, y)
    _, X_stop, _, y_stop = dr.early_stopping_split(X_train, y_train)
    truth = y_stop.values
    probabilities = []
    for seed in seeds:
        network, _, _ = dr.train_network(dr.build_conv_network, X, y, seed=seed, verbose=False)
        probabilities.append(dr.predict_probabilities(network, X_stop))
        print(f"seed {seed} alone: {(probabilities[-1].argmax(axis=1) == truth).mean():.4f}")
    probabilities = np.stack(probabilities)                       # networks × images × 10

    by_vote, ties = vote(probabilities)
    by_average = probabilities.mean(axis=0).argmax(axis=1)

    vote_right, average_right = by_vote == truth, by_average == truth
    print(f"votes: {vote_right.mean():.4f} ({(~vote_right).sum()} wrong, {ties} ties)")
    print(f"averaged probabilities: {average_right.mean():.4f} ({(~average_right).sum()} wrong)")
    print(f"the two differ on {(by_vote != by_average).sum()} images: votes right on {(vote_right & ~average_right).sum()}, "
          f"probabilities right on {(average_right & ~vote_right).sum()}")
    return probabilities, truth


def judge_voting(groups=5, size=5):
    # The voting model on the held-out set: `groups` voting models of `size` CNNs each, with their own seeds
    # (0-4, 5-9, ...), against the single CNN's mean over seeds 0-4 (the first group's members).
    # Keep rule: +0.3 points; check the disagreements (about 69 between two single CNNs).
    X, y = dr.get_X_y(dr.get_data_from_csv())
    X = dr.divide_by_255(X)
    _, X_test, _, y_test = dr.split_data(X, y)
    truth = y_test.values
    rows, single_accuracies = [], []
    for group in range(groups):
        seeds = list(range(group * size, (group + 1) * size))
        probabilities = np.stack([dr.predict_probabilities(
            dr.train_network(dr.build_conv_network, X, y, seed=seed, verbose=False)[0], X_test) for seed in seeds])
        if group == 0:
            single_accuracies = [(p.argmax(axis=1) == truth).mean() for p in probabilities]
        by_vote, ties = vote(probabilities)
        vote_right, first_right = by_vote == truth, probabilities[0].argmax(axis=1) == truth
        rows.append({'seeds': f"{seeds[0]}-{seeds[-1]}", 'voting': vote_right.mean(), 'ties': ties,
                     'first seed alone': first_right.mean(),
                     'disagree with it': (vote_right != first_right).sum()})
        print(rows[-1])
    table = pd.DataFrame(rows).set_index('seeds')
    print(table.to_string(float_format='{:.4f}'.format))
    single_mean = np.mean(single_accuracies)
    gain = (table['voting'].mean() - single_mean) * 100
    print(f"single CNN mean (seeds 0-4) {single_mean:.4f}, voting mean {table['voting'].mean():.4f} "
          f"(sd {table['voting'].std():.4f}), gain {gain:+.2f} points (keep rule: at least +0.30)")
    return table


if __name__ == '__main__':
    # model, X_val, y_val = baseline()
    # plot_mistakes(confused_pairs(model, X_val, y_val))
    # compare_epoch_choices()
    # the decision run: 10 epochs, last epoch kept (the reference setup then)
    # table = compare_networks({'dense': dr.build_dense_network, 'cnn': dr.build_conv_network},
    #                          epochs=10, early_stopping=False)
    compare_networks({'cnn': dr.build_conv_network,
                      'cnn_augmented': (dr.build_conv_network, {'augmentation': dr.random_shift_rotate_zoom})})
