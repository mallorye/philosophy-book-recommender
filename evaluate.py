def make_popularity_recommender(df, catalog):
    ranked = df["parent_asin"].value_counts().index.tolist()   # most-rated first

    def recommend(input_asins):
        return [a for a in ranked if a not in input_asins][:10]

    return recommend