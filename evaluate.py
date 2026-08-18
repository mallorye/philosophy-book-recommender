#---------- harness ----------
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