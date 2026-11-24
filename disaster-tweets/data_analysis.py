"""Step 1: look at the data before any model (plan.md, step 1).

Prints the tables and saves two plots to plots/. Run from anywhere:
    venv/Scripts/python.exe data_analysis.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "kaggle" / "input" / "nlp-getting-started"
PLOTS = HERE / "plots"

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 110)

URL = re.compile(r"https?://\S+")


def section(title):
    print(f"\n===== {title} =====")


def rate(s):
    """Disaster rate as a percentage, rounded."""
    return round(100 * s.mean(), 1)


def text_features(train):
    section("1. Text")
    t = train.text
    feats = pd.DataFrame({
        "target": train.target,
        "chars": t.str.len(),
        "words": t.str.split().str.len(),
        "has URL": t.str.contains(URL),
        "has @mention": t.str.contains(r"@\w"),
        "has #hashtag": t.str.contains(r"#\w"),
        "has HTML entity": t.str.contains(r"&\w+;"),
        "has line break": t.str.contains("\n"),
    })
    print("Median length by class:")
    print(feats.groupby("target")[["chars", "words"]].median())

    rows = []
    for col in [c for c in feats.columns if c.startswith("has ")]:
        with_it = feats[feats[col]]
        rows.append({"feature": col, "tweets": len(with_it), "share %": rate(feats[col]),
                     "disaster rate with %": rate(with_it.target),
                     "disaster rate without %": rate(feats[~feats[col]].target)})
    print(f"\nOverall disaster rate: {rate(train.target)}%")
    print(pd.DataFrame(rows).to_string(index=False))

    entities = t.str.findall(r"&\w+;").explode().value_counts()
    print("\nHTML entities (written as text, e.g. &amp; for &):", entities.to_dict())

    for label, name in [(1, "disaster"), (0, "not a disaster")]:
        print(f"\nExamples, {name}:")
        print(train[train.target == label].sample(5, random_state=0)[["keyword", "text"]].to_string(index=False))

    # plot: length by class
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = range(0, 165, 5)
    for label, name, color in [(0, "not a disaster", "#4C78A8"), (1, "disaster", "#E45756")]:
        ax.hist(feats.chars[feats.target == label], bins=bins, alpha=0.6, label=name, color=color)
    ax.set_xlabel("tweet length (characters)")
    ax.set_ylabel("tweets")
    ax.set_title("Tweet length by class")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS / "length_by_class.png", dpi=120)
    plt.close(fig)


def keyword(train):
    section("2. keyword")
    kw = train.keyword.str.replace("%20", " ")
    print(f"{kw.nunique()} values, {train.keyword.isna().sum()} missing ({rate(train.keyword.isna())}%), "
          f"{train.keyword.str.contains('%20', na=False).sum()} tweets with %20 for a space")
    per = train.assign(keyword=kw).groupby("keyword").target.agg(tweets="size", rate="mean")
    per["rate"] = (100 * per.rate).round(1)
    print(f"Tweets per keyword: {per.tweets.min()} to {per.tweets.max()}, median {per.tweets.median():.0f}")
    print(f"Disaster rate without a keyword: {rate(train.target[train.keyword.isna()])}%")

    print("\nNearly always a disaster (highest rate):")
    print(per.sort_values("rate", ascending=False).head(10).to_string())
    print("\nNearly never a disaster (lowest rate):")
    print(per.sort_values("rate").head(10).to_string())
    print("\nMixed (closest to 50%): the keyword alone can't decide")
    print(per.assign(d=(per.rate - 50).abs()).sort_values("d").head(10).drop(columns="d").to_string())

    # how much of the data has a keyword that nearly decides the label
    weighted = per.loc[kw.dropna()].rate
    print(f"\nTweets whose keyword's rate is >= 90%: {(weighted >= 90).sum()}, <= 10%: {(weighted <= 10).sum()}, "
          f"between: {((weighted > 10) & (weighted < 90)).sum()}")
    print("(rates computed on all of train: a model must learn them inside each fold)")

    for word in ["ablaze", "fire"]:
        rows = train[kw == word]
        print(f"\n'{word}': {len(rows)} tweets, disaster rate {rate(rows.target)}%")
        print(rows.sample(4, random_state=1)[["text", "target"]].to_string(index=False))

    # plot: distribution of per-keyword disaster rates
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per.rate, bins=range(0, 105, 5), color="#4C78A8", edgecolor="white")
    ax.set_xlabel("disaster rate of the keyword's tweets (%)")
    ax.set_ylabel("keywords")
    ax.set_title(f"How decisive each of the {len(per)} keywords is")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS / "keyword_rates.png", dpi=120)
    plt.close(fig)


def location(train):
    section("3. location")
    loc = train.location
    print(f"{loc.nunique()} values, {loc.isna().sum()} missing ({rate(loc.isna())}%)")
    print(f"Disaster rate with a location: {rate(train.target[loc.notna()])}%, "
          f"without: {rate(train.target[loc.isna()])}%")
    top = train.groupby("location").target.agg(tweets="size", rate="mean").sort_values("tweets", ascending=False)
    top["rate"] = (100 * top.rate).round(1)
    print("\nMost common locations (the same place written many ways):")
    print(top.head(15).to_string())
    print(f"\nLocations used by a single tweet: {(top.tweets == 1).sum()} of {len(top)}")


def normalise(text):
    """Text with URLs removed, lower case, spaces collapsed: retweets differ only by their t.co link."""
    return " ".join(URL.sub("", text).lower().split())


def duplicates(train):
    section("4. Duplicates")
    for name, key in [("identical text", train.text), ("same text without URLs, case, spaces", train.text.map(normalise))]:
        g = train.groupby(key).target.agg(["size", "nunique"])
        dup = g[g["size"] > 1]
        print(f"{name}: {len(dup)} texts appear more than once ({dup['size'].sum()} rows), "
              f"{(dup['nunique'] > 1).sum()} with conflicting labels ({dup.loc[dup['nunique'] > 1, 'size'].sum()} rows)")

    key = train.text.map(normalise)
    g = train.groupby(key).target.agg(["size", "nunique", "mean"])
    conflicting = g[g["nunique"] > 1].sort_values("size", ascending=False)
    print("\nConflicting groups, largest first (mean = share labelled 1):")
    print(conflicting.head(8).rename_axis("normalised text").to_string())


def train_vs_test(train, test):
    section("5. Train vs test")
    for name, df in [("train", train), ("test", test)]:
        print(f"{name}: {len(df)} tweets, median {df.text.str.len().median():.0f} characters, "
              f"keyword missing {rate(df.keyword.isna())}%, location missing {rate(df.location.isna())}%, "
              f"URL {rate(df.text.str.contains(URL))}%")
    tr_kw, te_kw = set(train.keyword.dropna()), set(test.keyword.dropna())
    print(f"keywords only in train: {len(tr_kw - te_kw)}, only in test: {len(te_kw - tr_kw)}")
    overlap = set(train.text.map(normalise)) & set(test.text.map(normalise))
    print(f"test tweets whose normalised text is also in train: "
          f"{test.text.map(normalise).isin(overlap).sum()}")


def main():
    PLOTS.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train {train.shape}, test {test.shape}")
    text_features(train)
    keyword(train)
    location(train)
    duplicates(train)
    train_vs_test(train, test)
    print(f"\nPlots saved to {PLOTS}")


if __name__ == "__main__":
    main()
