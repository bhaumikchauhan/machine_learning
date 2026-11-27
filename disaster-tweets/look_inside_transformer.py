"""Transformer step 4: look inside the BERT baseline (plan_transformer.md, step 4).

1. Where BERT and TF-IDF disagree, per class, out-of-fold on the same grouped folds.
2. BERT's mistakes left: the most confident ones, and how many the conflicting groups force.
3. The cut-off: F1 against the cut-off on BERT's out-of-fold probabilities (saved to plots/cutoff_bert.png).
4. Fold 0: hard for BERT for the same reason as for TF-IDF?
Also: how confident each model is (it matters for a blend, step 5e).

Reading only: nothing here changes either model. BERT's probabilities come from oof_bert_all_fold.csv
(step 3); TF-IDF's are recomputed (the final TF-IDF model is the baseline, seconds to fit).
    venv/Scripts/python.exe look_inside_transformer.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import PredefinedSplit, cross_val_predict

import data_analysis
from disaster_tweets import build_pipeline, get_tfidf_base_line
from look_inside import load

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 110)

TFIDF_FOLD_F1 = [0.7247, 0.7516, 0.7490, 0.7555, 0.7571]
BERT_FOLD_F1 = [0.7816, 0.7964, 0.8061, 0.7997, 0.7920]


def with_probabilities():
    df = load()
    cv = PredefinedSplit(test_fold=df["fold"])
    df["p_tfidf"] = cross_val_predict(build_pipeline(get_tfidf_base_line()), df.text, df.target,
                                      cv=cv, method="predict_proba")[:, 1]
    bert = pd.read_csv(data_analysis.HERE / "oof_bert_all_fold.csv")
    df = df.merge(bert[["id", "p"]].rename(columns={"p": "p_bert"}), on="id", how="inner", validate="one_to_one")
    assert len(df) == 7613
    # both sets of probabilities reproduce the logged per-fold F1
    for name, p, expected in [("tfidf", "p_tfidf", TFIDF_FOLD_F1), ("bert", "p_bert", BERT_FOLD_F1)]:
        got = [f1_score(g.target, g[p] >= 0.5) for _, g in df.groupby("fold")]
        assert np.allclose(np.round(got, 4), expected), (name, got)
    df["pred_tfidf"] = (df.p_tfidf >= 0.5).astype(int)
    df["pred_bert"] = (df.p_bert >= 0.5).astype(int)
    df["right_tfidf"] = df.pred_tfidf == df.target
    df["right_bert"] = df.pred_bert == df.target
    return df


def disagreements(df, n=10):
    data_analysis.section("1. Where the two models disagree (out-of-fold, same folds)")
    table = pd.crosstab([df.target.rename("label")], [df.right_bert.rename("BERT right"), df.right_tfidf.rename("TF-IDF right")])
    print(table.to_string())
    print(f"\nagree on the prediction: {(df.pred_bert == df.pred_tfidf).mean():.1%}; "
          f"correlation of the probabilities: {np.corrcoef(df.p_bert, df.p_tfidf)[0, 1]:.2f}")

    cols = ["target", "p_bert", "p_tfidf", "text"]
    only_bert = df[df.right_bert & ~df.right_tfidf]
    only_tfidf = df[~df.right_bert & df.right_tfidf]
    print(f"\nBERT right, TF-IDF wrong: {len(only_bert)}; TF-IDF right, BERT wrong: {len(only_tfidf)}")
    for label, name in [(0, "not disaster (TF-IDF's false alarms that BERT fixes)"), (1, "disaster (TF-IDF's misses that BERT fixes)")]:
        rows = only_bert[only_bert.target == label]
        rows = rows.assign(gap=(rows.p_bert - rows.p_tfidf).abs()).sort_values("gap", ascending=False)
        print(f"\n  BERT right, TF-IDF wrong, label {label}, {name}: {len(rows)}; the largest gaps:")
        print(rows.head(n)[cols].round(3).to_string(index=False))
    for label in [0, 1]:
        rows = only_tfidf[only_tfidf.target == label]
        rows = rows.assign(gap=(rows.p_bert - rows.p_tfidf).abs()).sort_values("gap", ascending=False)
        print(f"\n  TF-IDF right, BERT wrong, label {label}: {len(rows)}; the largest gaps:")
        print(rows.head(n)[cols].round(3).to_string(index=False))

    # TF-IDF step 4's most confident false alarms: the metaphors and the news style
    tfidf_fp = df[(df.pred_tfidf == 1) & (df.target == 0)].sort_values("p_tfidf", ascending=False).head(20)
    print(f"\nTF-IDF's 20 most confident false alarms: BERT also says disaster for {(tfidf_fp.pred_bert == 1).sum()}")
    print(tfidf_fp[cols].round(3).to_string(index=False))


def forced_errors(df):
    """Inside a group of (normalised) identical tweets, every copy gets about the same prediction,
    so at least the minority label's copies are wrong whatever the model."""
    key = df.text.map(data_analysis.normalise)
    groups = df[df.conflicting].groupby(key[df.conflicting]).target
    return int(groups.apply(lambda s: min(s.sum(), len(s) - s.sum())).sum())


def mistakes(df, n=10):
    data_analysis.section("2. BERT's mistakes left")
    for name, pred in [("BERT", df.pred_bert), ("TF-IDF", df.pred_tfidf)]:
        fp = ((pred == 1) & (df.target == 0)).sum()
        fn = ((pred == 0) & (df.target == 1)).sum()
        wrong = pred != df.target
        print(f"{name:7s} F1 {f1_score(df.target, pred):.4f}  precision {precision_score(df.target, pred):.3f}  "
              f"recall {recall_score(df.target, pred):.3f}  false positives {fp}, false negatives {fn}; "
              f"mistakes in a conflicting group: {(wrong & df.conflicting).sum()} of {wrong.sum()}")
    print(f"conflicting groups: {df.conflicting.sum()} tweets; at least {forced_errors(df)} of them are wrong for any model "
          f"(the minority label of each group)")

    cols = ["p_bert", "p_tfidf", "conflicting", "keyword", "text"]
    fp = df[(df.pred_bert == 1) & (df.target == 0)].sort_values("p_bert", ascending=False)
    fn = df[(df.pred_bert == 0) & (df.target == 1)].sort_values("p_bert")
    print(f"\nBERT's most confident false positives (labelled 0, BERT says disaster):")
    print(fp.head(n)[cols].round(3).to_string(index=False))
    print(f"\nBERT's most confident false negatives (labelled 1, BERT says not):")
    print(fn.head(n)[cols].round(3).to_string(index=False))


def confidence(df):
    data_analysis.section("How confident each model is")
    bins = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
    labels = ["<0.05", "0.05-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-0.95", ">=0.95"]
    rows = {}
    for name, p in [("BERT", df.p_bert), ("TF-IDF", df.p_tfidf)]:
        cut = pd.cut(p, bins, labels=labels, right=False)
        rows[(name, "share")] = cut.value_counts(normalize=True).reindex(labels)
        rows[(name, "disaster rate")] = df.target.groupby(cut, observed=False).mean().reindex(labels)
    print((pd.DataFrame(rows) * 100).round(1).to_string())
    print("(share = % of tweets in the probability band; disaster rate = % of them labelled 1: "
          "a calibrated model's rate sits inside its band)")


def cutoff(df):
    data_analysis.section("3. The cut-off (BERT, reading only)")
    cuts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for c in cuts:
        pred = (df.p_bert >= c).astype(int)
        rows.append({"cut-off": c, "F1": f1_score(df.target, pred),
                     "precision": precision_score(df.target, pred), "recall": recall_score(df.target, pred)})
    table = pd.DataFrame(rows).set_index("cut-off")
    best = table.F1.idxmax()
    print(table.loc[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]].round(4).to_string())
    print(f"\nBest cut-off over all out-of-fold predictions: {best} (F1 {table.F1[best]:.4f}, at 0.5: {table.F1[0.5]:.4f})")
    in_band = table[table.F1 >= table.F1[0.5] - 0.005].index
    print(f"F1 within 0.005 of the 0.5 value from cut-off {in_band.min()} to {in_band.max()}")

    print("\nPer fold, F1 at 0.5 against the cut-off that's best in that fold:")
    per = []
    for k, g in df.groupby("fold"):
        f1s = {c: f1_score(g.target, (g.p_bert >= c).astype(int)) for c in cuts}
        b = max(f1s, key=f1s.get)
        per.append({"fold": k, "F1 at 0.5": f1s[0.5], "best cut-off": b, "F1 at best": f1s[b]})
    print(pd.DataFrame(per).round(4).to_string(index=False))
    print("(each fold's best is chosen on its own validation tweets, so it's optimistic)")

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, color in [("F1", "#E45756"), ("precision", "#4C78A8"), ("recall", "#54A24B")]:
        ax.plot(table.index, table[col], color=color, linewidth=2, label=col)
    ax.axvline(0.5, color="#888888", linewidth=1, linestyle="--")
    ax.axvline(best, color="#E45756", linewidth=1, linestyle=":")
    ax.text(0.5, 0.52, " default 0.5", color="#555555", fontsize=9)
    ax.text(best, 0.48, f" best {best}", color="#E45756", fontsize=9)
    ax.set_xlabel("cut-off (say disaster if BERT's probability ≥ cut-off)")
    ax.set_ylabel("score (out-of-fold)")
    ax.set_title("BERT: F1, precision and recall against the cut-off")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(data_analysis.PLOTS / "cutoff_bert.png", dpi=120)
    plt.close(fig)


def hard_fold(df):
    data_analysis.section("4. Fold 0: hard for both models for the same reason?")
    per = df.groupby("fold").agg(
        acc_tfidf=("right_tfidf", "mean"), acc_bert=("right_bert", "mean"),
        both_wrong=("right_tfidf", lambda s: (~s & ~df.loc[s.index, "right_bert"]).mean()),
        only_tfidf_wrong=("right_tfidf", lambda s: (~s & df.loc[s.index, "right_bert"]).mean()),
        only_bert_wrong=("right_tfidf", lambda s: (s & ~df.loc[s.index, "right_bert"]).mean()),
        conflicting=("conflicting", "mean"))
    per["f1_tfidf"] = TFIDF_FOLD_F1
    per["f1_bert"] = BERT_FOLD_F1
    print((per * 100).round(1).to_string())
    print("(all in %: accuracy per model, then the share of the fold's tweets wrong for both / only one model)")

    wrong_tfidf = ~df.right_tfidf
    wrong_bert = ~df.right_bert
    print(f"\nTweets TF-IDF gets wrong: {wrong_tfidf.sum()}; BERT gets {(wrong_tfidf & wrong_bert).sum()} of them wrong too "
          f"({(wrong_tfidf & wrong_bert).sum() / wrong_tfidf.sum():.0%}); BERT's own mistakes: {wrong_bert.sum()}")


def main():
    df = with_probabilities()
    disagreements(df)
    mistakes(df)
    confidence(df)
    cutoff(df)
    hard_fold(df)
    print(f"\nPlot saved to {data_analysis.PLOTS / 'cutoff_bert.png'}")


if __name__ == "__main__":
    main()
