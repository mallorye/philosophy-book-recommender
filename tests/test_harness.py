import pandas as pd

from evaluate import get_eligible_users, hit_at_k, split_user


def make_df():
    return pd.DataFrame({
        "user_id":     ["u1"]*3 + ["u2"]*2 + ["u3"]*4,
        "parent_asin": ["a","b","c",  "a","d",  "b","c","d","e"],
        "rating":      [5,4,3,  5,5,  2,5,4,5],
        "timestamp":   [3,2,1,  1,2,  1,4,2,3],
    })


def test_eligible_users_threshold():
    assert set(get_eligible_users(make_df())) == {"u1", "u3"}   # >=3 ratings


def test_split_holds_out_highest_then_latest():
    u3 = make_df().query("user_id == 'u3'")
    inputs, held = split_user(u3)
    assert held == "c"            # rating 5 ties between c and e; c is more recent (ts 4 > 3)
    assert set(inputs) == {"b", "d", "e"}


def test_hit_at_k():
    assert hit_at_k(["x", "y", "z"], "y", k=3) is True
    assert hit_at_k(["x", "y", "z"], "z", k=2) is False