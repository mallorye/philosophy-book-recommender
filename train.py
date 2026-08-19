import json
import pickle

import pandas as pd

import wandb
from evaluate import evaluate
from models import (
    build_similarity,
    make_item_item_recommender,
    make_popularity_recommender,
)

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





# ---------- entry point ----------

def main():
    models = {
        "popularity": make_popularity_recommender,
        "item_item": make_item_item_recommender,
    }

    for model_name, factory in models.items():
        run = wandb.init(
            project="book-recommender",
            entity="malloryberg-university-of-denver",
            job_type="train",
            name=model_name,                    # readable run names in the UI
        )
        df, catalog = load_data_tracked(run)

        if model_name == "item_item":
            sim = build_similarity(df)
            recommend_fn = factory(df, catalog, sim=sim)
        else:
            recommend_fn = factory(df, catalog)
        score, n = evaluate(recommend_fn, df)

        run.config.update({
            "model": model_name,
            "k": 10,
            "split": "hold-out-1, highest rating, recency tiebreak",
            "dataset": "philosophy-dataset:v2",
        })
        run.log({"hit_rate_at_10": score, "n_eval_users": n})
        print(f"{model_name}: hit@10 = {score:.4f} (n={n})")

        if model_name == "item_item":
            model_payload = {
                "sim_matrix": sim,
                "popularity_ranked": df["parent_asin"].value_counts().index.tolist(),
                "k_default": 10,
            }
            with open("model.pkl", "wb") as f:
                pickle.dump(model_payload, f)

            model_art = wandb.Artifact(
                name="philosophy-recommender", type="model",
                metadata={"algorithm": "item-item CF, Pearson, min_periods=2",
                          "hit_rate_at_10": score, "n_eval_users": n,
                          "trained_on": "philosophy-dataset:v2"})
            model_art.add_file("model.pkl")
            run.log_artifact(model_art)

        run.finish()

      


if __name__ == "__main__":
    main()