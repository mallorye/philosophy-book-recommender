# app/main.py
import json
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import string
import boto3
import wandb
from fastapi import FastAPI
from pydantic import BaseModel

# ---------- request/response shapes (the contract, enforced) ----------

class PredictRequest(BaseModel):
    favorites: list[str]
    k: int = 10

class Recommendation(BaseModel):
    asin: str
    title: str
    author: str | None = None
    average_rating: float | None = None

class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    recommendations: list[Recommendation]
    unmatched_favorites: list[str]
    fallback_used: bool
    latency_ms: int

# ---------- app state, loaded once at startup ----------

STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    api = wandb.Api()
    model_art = api.artifact(
        "malloryberg-university-of-denver/book-recommender/philosophy-recommender:production")
    model_dir = model_art.download()
    with open(f"{model_dir}/model.pkl", "rb") as f:
        payload = pickle.load(f)

    data_art = api.artifact(
        "malloryberg-university-of-denver/book-recommender/philosophy-dataset:v2")
    data_dir = data_art.download()
    with open(f"{data_dir}/catalog.json") as f:
        catalog = json.load(f)

    STATE["sim"] = payload["sim_matrix"]
    STATE["popularity_ranked"] = payload["popularity_ranked"]
    STATE["catalog"] = catalog
    STATE["model_version"] = f"philosophy-recommender:{model_art.version}"
    # normalized title -> asin, built once:
    STATE["title_index"] = build_title_index(catalog)

    STATE["dynamo"] = boto3.resource(
        "dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    STATE["pred_table"] = STATE["dynamo"].Table("predictions")

    yield
    STATE.clear()

app = FastAPI(lifespan=lifespan)

# ----------  title resolution ----------

def normalize(title: str) -> str:
    """lowercase, strip punctuation/whitespace — """
    title = title.lower()
    title = title.strip()
    title = title.translate(str.maketrans('', '', string.punctuation))
    return title

def build_title_index(catalog: dict) -> dict:
    """{normalize(info['title']): asin} for every catalog entry"""
    return {normalize(info['title']): asin for asin, info in catalog.items()}


def resolve_titles(favorites):
    matched, unmatched = [], []
    index = STATE["title_index"]          # normalized title -> asin
    for title in favorites:
        q = normalize(title)
        asin = index.get(q)               # exact first
        if not asin and len(q) >= 4:      # then substring
            for cat_title, cat_asin in index.items():
                if q in cat_title:
                    asin = cat_asin
                    break
        if asin:
            matched.append(asin)
        else:
            unmatched.append(title)
    return matched, unmatched
    

# ---------- recommendation from the pickle payload ----------

def recommend(input_asins: list[str], k: int) -> tuple[list[str], bool]:
    """Returns (recommended_asins, fallback_used).
    Same logic as make_item_item_recommender's inner function, except the
    cold-start fallback reads STATE['popularity_ranked'] instead of df.
    Uses STATE['sim']."""
    ...
    sim = STATE["sim"] 
    known = [a for a in input_asins if a in STATE["sim"].columns]
    if not known:                       # cold-start fallback: popularity
        ranked = STATE["popularity_ranked"]
        return [a for a in ranked if a not in input_asins][:k], True

    scores = sim[known].sum(axis=1, skipna=True)

    scores = scores.drop(labels=[a for a in input_asins if a in scores.index])
    return scores.sort_values(ascending=False).head(k).index.tolist(), False
# ---------- endpoints (complete) ----------

@app.get("/health")
def health():
    return {"status": "ok",
            "model_version": STATE["model_version"],
            "catalog_size": len(STATE["catalog"])}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.time()
    request_id = str(uuid.uuid4())

    matched, unmatched = resolve_titles(req.favorites)
    rec_asins, fallback = recommend(matched, req.k)
    latency_ms = int((time.time() - t0) * 1000)

    recs = []
    for asin in rec_asins:
        info = STATE["catalog"].get(asin, {})
        recs.append(Recommendation(
            asin=asin, title=info.get("title", "Unknown"),
            author=info.get("author"),
            average_rating=info.get("average_rating")))

    try:
        STATE["pred_table"].put_item(Item={
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": STATE["model_version"],
            "input_titles": req.favorites,
            "matched_asins": matched,
            "recommended_asins": rec_asins,
            "fallback_used": fallback,
            "latency_ms": latency_ms,
        })
    except Exception as e:
        print(f"WARN: prediction log failed: {e}")   # serving > logging today

    return PredictResponse(
        request_id=request_id, model_version=STATE["model_version"],
        recommendations=recs, unmatched_favorites=unmatched,
        fallback_used=fallback, latency_ms=latency_ms)