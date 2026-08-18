import random


def make_random_recommender(catalog):
    all_asins = list(catalog.keys())

    def recommend(input_asins):
        candidates = [a for a in all_asins if a not in input_asins]
        return random.sample(candidates, 10)

    return recommend


def make_popularity_recommender(df, catalog):
    ranked = df["parent_asin"].value_counts().index.tolist()   # most-rated first

    def recommend(input_asins):
        return [a for a in ranked if a not in input_asins][:10]

    return recommend

def build_similarity(df):
    """user×item pivot -> item×item Pearson similarity (≥2 co-raters)."""
    mat = df.pivot_table(index="user_id", columns="parent_asin", values="rating")
    return mat.corr(min_periods=2)

def make_item_item_recommender(df, catalog, sim=None):
    if sim is None:
        sim = build_similarity(df)
  
    # item × item cosine-ish similarity via pandas corr on co-raters

    def recommend(input_asins, k=10):
        known = [a for a in input_asins if a in sim.columns]
        if not known:                       # cold-start fallback: popularity
            ranked = df["parent_asin"].value_counts().index.tolist()
            return [a for a in ranked if a not in input_asins][:k]
        scores = sim[known].sum(axis=1, skipna=True)
        scores = scores.drop(labels=[a for a in input_asins if a in scores.index])
        return scores.sort_values(ascending=False).head(k).index.tolist()

    return recommend