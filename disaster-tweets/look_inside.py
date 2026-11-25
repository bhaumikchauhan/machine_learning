"""Step 4: look inside the baseline (plan.md, step 4).

1. The words: the largest positive and negative weights of TF-IDF + logistic regression.
2. The mistakes: the most confident false positives and false negatives, out-of-fold on the grouped folds.
3. The cut-off: F1, precision and recall against the probability cut-off (saved to plots/cutoff.png).
Also: why fold 0 is the hard one.

Reading only: nothing here changes the model. Run from anywhere:
    venv/Scripts/python.exe look_inside.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import PredefinedSplit, cross_val_predict

import data_analysis
from disaster_tweets import FOLDS_CSV, build_pipeline, get_train_data_from_csv

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)


def load():
    df = get_train_data_from_csv().merge(pd.read_csv(FOLDS_CSV), on="id")
    key = df.text.map(data_analysis.normalise)
    sizes = key.map(key.value_counts())
    labels_per_group = df.groupby(key).target.transform("nunique")
    df["repeated"] = sizes > 1
    df["conflicting"] = labels_per_group > 1
    return df


def words(df, n=20):
    data_analysis.section("1. The words (fitted on all of train, for reading)")
    model = build_pipeline().fit(df.text, df.target)
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    weights = model.named_steps["lr"].coef_[0]
    counts = pd.Series((model.named_steps["tfidf"].transform(df.text) > 0).sum(axis=0).A1, index=vocab)
    table = pd.DataFrame({"weight": weights.round(2), "tweets": counts.values}, index=vocab)
    print(f"{len(vocab):,} words (columns) for {len(df):,} tweets")
    side = pd.concat([
        table.nlargest(n, "weight").reset_index(names="towards disaster"),
        table.nsmallest(n, "weight").reset_index(names="towards not disaster"),
    ], axis=1)
    print(side.to_string(index=False))
    print("(tweets = how many training tweets contain the word)")


def out_of_fold(df):
    """Each tweet's probability from the model that didn't train on its fold."""
    cv = PredefinedSplit(test_fold=df["fold"])
    return cross_val_predict(build_pipeline(), df.text, df.target, cv=cv, method="predict_proba")[:, 1]


def mistakes(df, n=10):
    data_analysis.section("2. The mistakes (out-of-fold, grouped folds)")
    pred = (df.p >= 0.5).astype(int)
    fp = df[(pred == 1) & (df.target == 0)]
    fn = df[(pred == 0) & (df.target == 1)]
    print(f"F1 at 0.5 over all out-of-fold predictions: {f1_score(df.target, pred):.4f}")
    print(f"false positives: {len(fp)}, false negatives: {len(fn)}")
    for name, rows in [("false positives", fp), ("false negatives", fn)]:
        print(f"  {name}: {rows.conflicting.mean():.1%} in a conflicting group, "
              f"{rows.repeated.mean():.1%} repeated (all tweets: {df.conflicting.mean():.1%}, {df.repeated.mean():.1%})")

    cols = ["p", "keyword", "text"]
    print(f"\nMost confident false positives (labelled 0, model says disaster):")
    print(fp.sort_values("p", ascending=False).head(n)[cols].round(3).to_string(index=False))
    print(f"\nMost confident false negatives (labelled 1, model says not):")
    print(fn.sort_values("p").head(n)[cols].round(3).to_string(index=False))


def cutoff(df):
    data_analysis.section("3. The cut-off")
    cuts = np.round(np.arange(0.20, 0.71, 0.01), 2)
    rows = []
    for c in cuts:
        pred = (df.p >= c).astype(int)
        rows.append({"cut-off": c, "F1": f1_score(df.target, pred),
                     "precision": precision_score(df.target, pred), "recall": recall_score(df.target, pred)})
    table = pd.DataFrame(rows).set_index("cut-off")
    best = table.F1.idxmax()
    print(table.loc[[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]].round(4).to_string())
    print(f"\nBest cut-off over all out-of-fold predictions: {best} (F1 {table.F1[best]:.4f}, at 0.5: {table.F1[0.5]:.4f})")

    print("\nPer fold, F1 at 0.5 against the cut-off that's best in that fold:")
    per = []
    for k, g in df.groupby("fold"):
        f1s = {c: f1_score(g.target, (g.p >= c).astype(int)) for c in cuts}
        b = max(f1s, key=f1s.get)
        per.append({"fold": k, "F1 at 0.5": f1s[0.5], "best cut-off": b, "F1 at best": f1s[b]})
    print(pd.DataFrame(per).round(4).to_string(index=False))
    print("(each fold's best is chosen on its own validation tweets, so it's optimistic: a kept cut-off must be chosen inside the folds, step 5g)")

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, color in [("F1", "#E45756"), ("precision", "#4C78A8"), ("recall", "#54A24B")]:
        ax.plot(table.index, table[col], color=color, linewidth=2, label=col)
    ax.axvline(0.5, color="#888888", linewidth=1, linestyle="--")
    ax.axvline(best, color="#E45756", linewidth=1, linestyle=":")
    ax.text(0.5, 0.42, " default 0.5", color="#555555", fontsize=9)
    ax.text(best, 0.38, f" best {best}", color="#E45756", fontsize=9)
    ax.set_xlabel("cut-off (say disaster if probability ≥ cut-off)")
    ax.set_ylabel("score (out-of-fold)")
    ax.set_title("F1, precision and recall against the cut-off")
    ax.set_ylim(0.35, 1.0)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(data_analysis.PLOTS / "cutoff.png", dpi=120)
    plt.close(fig)


def hard_fold(df):
    data_analysis.section("4. Why fold 0 is the hard one")
    per = df.assign(right=((df.p >= 0.5).astype(int) == df.target)).groupby("fold").agg(
        tweets=("id", "size"), disaster_pct=("target", "mean"), repeated_pct=("repeated", "mean"),
        conflicting_pct=("conflicting", "mean"), mean_p_when_1=("p", lambda s: s[df.loc[s.index, "target"] == 1].mean()),
        accuracy=("right", "mean"))
    for c in ["disaster_pct", "repeated_pct", "conflicting_pct"]:
        per[c] = (100 * per[c]).round(1)
    print(per.round(3).to_string())


def main():
    df = load()
    words(df)
    df["p"] = out_of_fold(df)
    mistakes(df)
    cutoff(df)
    hard_fold(df)
    print(f"\nPlot saved to {data_analysis.PLOTS / 'cutoff.png'}")


if __name__ == "__main__":
    main()
