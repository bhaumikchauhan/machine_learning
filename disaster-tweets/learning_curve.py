"""Step 6: learning curve of the final TF-IDF model (plan.md, step 6).

Trains on 10% ... 100% of each grouped fold's training part and scores on that fold's validation tweets.
Still rising at 100% means more tweets would help; flat means bag of words has reached its limit.
Saves plots/learning_curve.png. Run from anywhere:
    venv/Scripts/python.exe learning_curve.py
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, learning_curve

import data_analysis
from disaster_tweets import FOLDS_CSV, RANDOM_SEED, build_pipeline, get_tfidf_base_line

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    df = pd.read_csv(data_analysis.DATA / "train.csv").merge(pd.read_csv(FOLDS_CSV), on="id")
    sizes, train, val = learning_curve(
        build_pipeline(get_tfidf_base_line()), df.text, df.target,
        cv=PredefinedSplit(df["fold"]), scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True, random_state=RANDOM_SEED,  # a random subset of each training part, not its first rows
    )
    table = pd.DataFrame({
        "tweets": sizes,
        "train F1": train.mean(axis=1), "val F1": val.mean(axis=1), "val sd": val.std(axis=1),
        "gap": train.mean(axis=1) - val.mean(axis=1),
    }).round(4)
    data_analysis.section("Learning curve (grouped folds, mean of 5)")
    print(table.to_string(index=False))

    # the last doubling: half the tweets -> all of them, compared fold by fold
    half = int(np.argmin(np.abs(sizes - sizes[-1] / 2)))
    diff = val[-1] - val[half]
    print(f"\n{sizes[half]:,} -> {sizes[-1]:,} tweets (about a doubling): val F1 {val[half].mean():.4f} -> {val[-1].mean():.4f}, "
          f"gain {diff.mean():+.4f}, per fold {np.round(diff, 4).tolist()}")
    # straight line in log2(tweets) through the upper half: the gain per doubling
    upper = slice(half, None)
    slope = np.polyfit(np.log2(sizes[upper]), val.mean(axis=1)[upper], 1)[0]
    print(f"trend over the upper half: {slope:+.4f} F1 per doubling of the tweets")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for scores, name, color in [(train, "train", "#4C78A8"), (val, "validation", "#E45756")]:
        m, s = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, m, color=color, linewidth=2, marker="o", markersize=5, label=name)
        ax.fill_between(sizes, m - s, m + s, color=color, alpha=0.15, linewidth=0)
    ax.axhline(0.6011, color="#888888", linestyle="--", linewidth=1)
    ax.text(sizes[-1], 0.607, "always disaster 0.60", color="#555555", fontsize=9, ha="right")
    ax.set_xlabel("training tweets per fold")
    ax.set_ylabel("F1 (mean of 5 grouped folds, band = ±1 sd)")
    ax.set_title("Learning curve: TF-IDF + logistic regression")
    ax.set_ylim(0.55, 1.0)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(data_analysis.PLOTS / "learning_curve.png", dpi=120)
    plt.close(fig)
    print(f"\nPlot saved to {data_analysis.PLOTS / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
