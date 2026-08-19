# dashboard_app.py — Wayfinder Ops (monitoring dashboard, separate EC2)
#
# Reads ONLY from DynamoDB (predictions, feedback) + catalog.json for display/
# category joins. Never calls the backend — passive log observer.
#
# Metric definitions (also stated in the sidebar, and in the README):
#   Latency        — latency_ms per prediction over time (rolling median + p95)
#   Target drift   — distribution of recommended sub-genres over time; proxy for
#                    "distribution of predicted classes" (no classes in a
#                    recommender; sub-genre = tail of the catalog category path)
#   Live accuracy  — fraction of feedback events marked helpful (precision proxy),
#                    always shown with n
#
# Env vars:
#   AWS_REGION    (default us-east-1)
#   CATALOG_PATH  (default catalog.json; falls back to W&B artifact if missing)

import json
import os

import boto3
import pandas as pd
import streamlit as st

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
CATALOG_PATH = os.environ.get("CATALOG_PATH", "catalog.json")

st.set_page_config(page_title="Wayfinder Ops", page_icon="📈", layout="wide")
st.title("📈 Wayfinder Ops — model monitoring")

with st.sidebar:
    st.header("Metric definitions")
    st.markdown(
        "**Latency** — per-request `latency_ms`, rolling median and p95.\n\n"
        "**Target drift** — share of recommendations per *sub-genre* over time. "
        "A recommender has no predicted classes; sub-genre (from the catalog "
        "category path) is the stated proxy.\n\n"
        "**Live accuracy** — fraction of user feedback marked 👍 "
        "(a precision proxy), reported with sample size n."
    )
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()


# ---------- data loading ----------

def scan_all(table_name: str) -> list[dict]:
    """Full table scan with pagination. Fine at course scale (documented trade-off)."""
    table = boto3.resource("dynamodb", region_name=AWS_REGION).Table(table_name)
    items, resp = [], table.scan()
    items.extend(resp.get("Items", []))
    while "LastEvaluatedKey" in resp:
        resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
        items.extend(resp.get("Items", []))
    return items


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    df = pd.DataFrame(scan_all("predictions"))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["latency_ms"] = df["latency_ms"].astype(float)  # Decimal -> float
    return df.sort_values("timestamp")


@st.cache_data(ttl=60)
def load_feedback() -> pd.DataFrame:
    df = pd.DataFrame(scan_all("feedback"))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_catalog() -> dict:
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            return json.load(f)
    import wandb  # fallback: pull from the dataset artifact
    art = wandb.Api().artifact(
        "malloryberg-university-of-denver/book-recommender/philosophy-dataset:v2")
    d = art.download()
    with open(f"{d}/catalog.json") as f:
        return json.load(f)


def sub_genre(asin: str, catalog: dict) -> str:
    """Drift dimension: author. Category paths in this catalog rarely go
    below 'Philosophy' (94% of items), so author concentration is the
    stated drift proxy instead."""
    return (catalog.get(asin, {}) or {}).get("author") or "Unknown"


preds = load_predictions()
fb = load_feedback()
catalog = load_catalog()

if preds.empty:
    st.info("No predictions logged yet. Send a request through the frontend "
            "or curl /predict, then refresh.")
    st.stop()

# ---------- headline numbers ----------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total predictions", f"{len(preds):,}")
c2.metric("Median latency", f"{preds['latency_ms'].median():.0f} ms")
c3.metric("Fallback rate", f"{preds['fallback_used'].mean():.0%}")
if fb.empty:
    c4.metric("Live accuracy", "—", help="No feedback yet")
else:
    acc = fb["helpful"].mean()
    c4.metric("Live accuracy", f"{acc:.0%}", help=f"n = {len(fb)} feedback events")

st.divider()

# ---------- 1. latency over time ----------

st.subheader("Prediction latency over time")
lat = preds.set_index("timestamp")["latency_ms"]
window = max(5, len(lat) // 10)
chart_df = pd.DataFrame({
    "latency_ms": lat,
    "rolling_median": lat.rolling(window, min_periods=1).median(),
    "rolling_p95": lat.rolling(window, min_periods=1).quantile(0.95),
})
st.line_chart(chart_df)

# ---------- 2. target drift: sub-genre distribution over time ----------

st.subheader("Recommended sub-genre distribution (target-drift proxy)")
rows = []
for _, p in preds.iterrows():
    for asin in p["recommended_asins"]:
        rows.append({"timestamp": p["timestamp"], "sub_genre": sub_genre(asin, catalog)})
drift = pd.DataFrame(rows)
if drift.empty:
    st.info("No recommendations logged yet.")
else:
    drift["hour"] = drift["timestamp"].dt.floor("h")
    top = drift["sub_genre"].value_counts().head(8).index
    drift["sub_genre"] = drift["sub_genre"].where(drift["sub_genre"].isin(top), "Other")
    pivot = (drift.groupby(["hour", "sub_genre"]).size().unstack(fill_value=0))
    share = pivot.div(pivot.sum(axis=1), axis=0)          # shares, not counts
    if len(share) < 2:
        st.bar_chart(share.iloc[-1].sort_values(ascending=False))
    else:
        st.area_chart(share)
    st.caption("Share of recommended books per sub-genre, hourly. "
               "A stable mix = no drift; a shifting mix = the model's output "
               "distribution is moving.")

# ---------- 3. live accuracy from feedback ----------

st.subheader("Live accuracy (user feedback)")
if fb.empty:
    st.info("No feedback yet — 👍/👎 in the frontend feeds this chart.")
else:
    fbt = fb.sort_values("timestamp").set_index("timestamp")
    cum = fbt["helpful"].expanding().mean()
    st.line_chart(pd.DataFrame({"cumulative_accuracy": cum}))
    st.caption(f"Cumulative share of 👍 across {len(fb)} feedback events. "
               "Interpret with n in view — early values rest on few votes.")

    joined = fb.merge(
        preds[["request_id", "model_version"]], on="request_id", how="left")
    by_model = (joined.groupby("model_version")["helpful"]
                .agg(accuracy="mean", n="count"))
    st.dataframe(by_model)
