import json
import pandas as pd
import wandb
from evaluate import make_popularity_recommender
import random

# ---------- data ----------

def load_data_local():
    """Dev loading: reads the cached artifact files directly. No W&B, no run."""
    data_dir = "artifacts/philosophy-dataset:v2"
    df = pd.read_json(f"{data_dir}/philosophy_reviews_pruned.jsonl", lines=True)
    with open(f"{data_dir}/catalog.json") as f:
        catalog = json.load(f)
    return df, catalog


def load_data_tracked(run):
    """Real-run loading: same files, but declares lineage via use_artifact."""
    artifact = run.use_artifact("philosophy-dataset:latest")
    data_dir = artifact.download()
    df = pd.read_json(f"{data_dir}/philosophy_reviews_pruned.jsonl", lines=True)
    with open(f"{data_dir}/catalog.json") as f:
        catalog = json.load(f)
    return df, catalog

# ---------- harness ----------
# who can be tested
def get_eligible_users(df):
    """Users with >= 3 ratings: enough to split into inputs + one held out."""
    counts = df["user_id"].value_counts()
    return counts[counts >= 3].index.tolist()

#-------split user ------------
#one user's exam question
def split_user(user_df):
    """
    One user's ratings -> (input_asins, held_out_asin).
    """
    ranked = user_df.sort_values(["rating", "timestamp"], ascending=False)
    held_out = ranked.iloc[0]["parent_asin"]          # top row's book
    inputs = ranked.iloc[1:]["parent_asin"].tolist()  # every other row's book
    return inputs, held_out

#---------hit at k ---------------
def hit_at_k(recommendations, held_out_asin, k=10):
    """
    Did the hidden book appear in the top k?
    """
    return held_out_asin in recommendations[:k]

def evaluate(recommend_fn, df, k=10):
    hits = []
    for user_id in get_eligible_users(df):
        user_df = df[df["user_id"] == user_id]
        inputs, held_out = split_user(user_df)
        recs = recommend_fn(inputs)
        hits.append(hit_at_k(recs, held_out, k))
    return sum(hits) / len(hits), len(hits)   # score, n

#writing worlds worst recommender as a low baseline

def make_random_recommender(catalog):
    all_asins = list(catalog.keys())

    def recommend(input_asins):
        candidates = [a for a in all_asins if a not in input_asins]
        return random.sample(candidates, 10)

    return recommend


# ---------- entry point ----------

def main():
    run = wandb.init(
        project="book-recommender",
        entity="malloryberg-university-of-denver",
        job_type="train",
    )
    df, catalog = load_data_tracked(run)
    eligible = get_eligible_users(df)
    print(f"{len(df):,} reviews | {len(catalog):,} catalog | {len(eligible):,} eligible users")

    score, n = evaluate(make_popularity_recommender(df, catalog), df)
    run.config.update({"model": "popularity", "k": 10, "split": "hold-out-1, highest rating, recency tiebreak"})
    run.log({"hit_rate_at_10": score, "n_eval_users": n})

    run.finish()


if __name__ == "__main__":
    main()