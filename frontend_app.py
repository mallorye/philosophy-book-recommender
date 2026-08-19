# frontend_app.py — The Wisdom Wayfinder (user-facing Streamlit app)
#
# Contract dependencies:
#   POST {BACKEND_URL}/predict  {"favorites": [...], "k": int}
#     -> {request_id, model_version, recommendations[{asin,title,author,average_rating}],
#         unmatched_favorites, fallback_used, latency_ms}
#   DynamoDB table "feedback": PK request_id (S), SK asin (S), helpful (BOOL), timestamp (S)
#
# Env vars:
#   BACKEND_URL  (default http://localhost:8000)
#   AWS_REGION   (default us-east-1)

import os
from datetime import datetime, timezone

import boto3
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

st.set_page_config(page_title="The Wisdom Wayfinder", page_icon="📚")
st.title("📚 The Wisdom Wayfinder")
st.caption("A what-to-read-next recommender for philosophy. "
           "Enter a few books you love; get your next one.")


@st.cache_resource
def feedback_table():
    dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
    return dynamo.Table("feedback")


def send_feedback(request_id: str, asin: str, helpful: bool):
    try:
        feedback_table().put_item(Item={
            "request_id": request_id,
            "asin": asin,
            "helpful": helpful,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return True
    except Exception as e:
        st.warning(f"Feedback could not be saved: {e}")
        return False


# ---------- input form ----------

with st.form("favorites_form"):
    raw = st.text_area(
        "Your favorite philosophy books (one per line)",
        placeholder="Meditations\nThe Story of Philosophy",
        height=100,
    )
    k = st.slider("How many recommendations?", 1, 10, 5)
    submitted = st.form_submit_button("Find my next book")

if submitted:
    favorites = [line.strip() for line in raw.splitlines() if line.strip()]
    if not favorites:
        st.error("Enter at least one title.")
    else:
        try:
            resp = requests.post(
                f"{BACKEND_URL}/predict",
                json={"favorites": favorites, "k": k},
                timeout=30,
            )
            resp.raise_for_status()
            st.session_state["result"] = resp.json()
            st.session_state["feedback_given"] = {}
        except Exception as e:
            st.error(f"Could not reach the recommender: {e}")

# ---------- results (persist across reruns via session_state) ----------

result = st.session_state.get("result")
if result:
    if result["unmatched_favorites"]:
        st.info("Couldn't find in the catalog: "
                + ", ".join(result["unmatched_favorites"]))
    if result["fallback_used"]:
        st.info("None of your titles matched, so these are the most-read "
                "philosophy books overall.")

    st.subheader("Recommended for you")
    given = st.session_state.setdefault("feedback_given", {})

    for rec in result["recommendations"]:
        asin = rec["asin"]
        cols = st.columns([6, 1, 1])
        with cols[0]:
            line = f"**{rec['title']}**"
            if rec.get("author"):
                line += f" — {rec['author']}"
            if rec.get("average_rating") is not None:
                line += f"  ·  ★{rec['average_rating']}"
            st.markdown(line)
        with cols[1]:
            if st.button("👍", key=f"up_{asin}", disabled=asin in given):
                if send_feedback(result["request_id"], asin, True):
                    given[asin] = True
                    st.rerun()
        with cols[2]:
            if st.button("👎", key=f"down_{asin}", disabled=asin in given):
                if send_feedback(result["request_id"], asin, False):
                    given[asin] = False
                    st.rerun()

    if given:
        st.caption(f"Thanks — feedback recorded for {len(given)} "
                   f"book{'s' if len(given) != 1 else ''}.")

    st.caption(f"model {result['model_version']} · "
               f"{result['latency_ms']} ms · request {result['request_id'][:8]}…")
