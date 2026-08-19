# The Wisdom Wayfinder

*A what-to-read-next recommender for philosophy, built as a production-grade MLOps system.*

Built as the final project for COMP 4450 (MLOps), Summer 2026. The recommender is
deliberately simple; the point of the project is the production system around it:
experiment tracking, model registry, serving, logging, monitoring, CI/CD, and
cloud deployment.

**Live components (while the AWS Academy Learner Lab session is active):**
- User frontend (Streamlit): `http://<APP_EC2_IP>:8501`
- ML backend (FastAPI): `http://<APP_EC2_IP>:8000` (`/predict`, `/health`, `/docs`)
- Monitoring dashboard (Streamlit, separate EC2): `http://<OPS_EC2_IP>:8501`
- W&B project (experiments, dataset + model artifacts): https://wandb.ai/malloryberg-university-of-denver/book-recommender

## Architecture

```
                    ┌─────────────────┐
                    │  Weights & Biases│
                    │  dataset + model │
                    │  registry        │
                    └───────┬─────────┘
                            │ pull model:production at startup
        ┌───────────────────▼──────────────┐      ┌──────────────────────┐
        │  EC2 "wayfinder-app"             │      │  EC2 "wayfinder-ops" │
        │  ┌──────────┐   ┌────────────┐   │      │  ┌────────────────┐  │
        │  │ Streamlit │──▶│  FastAPI   │   │      │  │ Streamlit      │  │
        │  │ frontend  │   │  backend   │   │      │  │ dashboard      │  │
        │  └─────┬─────┘   └─────┬──────┘   │      │  └───────┬────────┘  │
        └────────┼───────────────┼──────────┘      └──────────┼───────────┘
                 │ feedback      │ prediction logs            │ reads only
                 ▼               ▼                            ▼
              ┌──────────────────────────────────────────────────┐
              │        DynamoDB: predictions, feedback            │
              └──────────────────────────────────────────────────┘
```

The monitoring dashboard runs on a **separate EC2 instance** and exchanges data
with the serving stack **only through DynamoDB** — it is a passive log observer,
never in the request path. Both instances authenticate to DynamoDB via the
Learner Lab IAM instance role (LabRole); no AWS credentials are stored anywhere.
The only secret in the system is the W&B API key, supplied as an environment
variable to the backend container.

## Data

**Source:** McAuley Lab Amazon Reviews 2023, Books category
([Hou et al., 2024](https://arxiv.org/abs/2403.03952);
https://amazon-reviews-2023.github.io/). Raw scale: 29.5M reviews across 4.4M
book items. The raw data is processed once, locally, in
`notebooks/01_data_prep.ipynb`; only the frozen outputs ever reach W&B or AWS.

**Scope:** the catalog is items whose Amazon category path contains
"philosoph" (case-insensitive, matched across all elements of the hierarchical
path). Adjacent religion/spirituality titles are out of scope — a deliberate
precision-over-recall boundary: within a scoped catalog every co-rating edge
connects topically pre-screened books, which is what keeps the collaborative
signal interpretable.

**Pipeline:** stream the 4.4M-item metadata file → philosophy filter →
17,112 candidate items → stream the 29.5M-review file → semi-join on
`parent_asin` (the documented canonical item key; per-edition `asin`s would
fragment one book's ratings across formats) → 53,518 philosophy reviews →
**iterated k-core pruning** (items ≥3 ratings, users ≥2 ratings, repeated until
stable — 6 passes) → **4,154 reviews, 736 items, 1,537 users**.

Two pruning rules, each with a mechanism: users need ≥2 ratings because a
single-rating user contributes zero co-occurrence pairs; items keep a *low*
floor (≥3) deliberately — within a domain-scoped catalog even thin co-reader
edges connect pre-screened books, and low-evidence items are precisely where a
recommender adds value over popularity ranking. Single-pass pruning overstates
the stable core by ~85% (7,674 vs 4,154 reviews); the notebook iterates to a
true k-core and asserts the invariants.

The frozen dataset (`philosophy_reviews_pruned.jsonl` + `catalog.json`) is
versioned as the W&B artifact `philosophy-dataset` (v2 = the corrected,
stable-core version; v1 is preserved as an honest record of the single-pass
mistake).

## Model

Two models, one shared evaluation harness, both tracked as W&B runs:

| model | hit-rate@10 | n (eval users) |
|---|---|---|
| popularity baseline | 0.0535 | 430 |
| **item–item CF (production)** | **0.1186** | 430 |
| (random calibration) | 0.0093 | 430 |

**Evaluation:** for each of the 430 users with ≥3 ratings, hold out their
highest-rated book (ties broken by most recent), feed the rest as "favorites,"
and check whether the held-out book appears in the top 10. No validation split
was used because no hyperparameter tuning was performed; each model faced the
fixed hold-out exactly once. The harness was calibrated against a random
recommender (0.0093 ≈ the 10/735 chance rate) before any real model was scored.

Absolute hit-rates are low, as is typical for top-k recommendation over
hundreds of candidates; what matters is the ratio — the collaborative model
improves on popularity by 2.2×, clearing the pre-registered noise band
(±0.02–0.04 at n=430).

**The model is deliberately minimal** — item–item collaborative filtering with
Pearson similarity over the user×item rating matrix, requiring ≥2 co-raters per
item pair (`min_periods=2`). Model quality was explicitly deprioritized in
favor of the production system; tuning, significance weighting, edition
dedup, and hybrid approaches are catalogued under future work.

**This system is collaborative, not content-based:** predictions derive from
co-reading behavior, not from any representation of the books' ideas. Topical
coherence is enforced by the catalog boundary (the philosophy filter), not
learned by the model — the same code would recommend restaurants if fed
restaurant ratings. Author-awareness is implicit in the collaborative signal
(philosophy reading is strongly author-anchored, and same-author books share
readerships); explicit author features are future work.

**Cold start / unknown titles:** title resolution is normalized substring
match against the catalog (first hit wins among editions; fuzzy matching is
future work). Unmatched favorites are skipped and reported back in the
response (`unmatched_favorites`); if nothing matches, the API falls back to
the popularity ranking and says so (`fallback_used: true`).

**Registry:** the trained model (similarity matrix + popularity fallback
ranking) is the W&B artifact `philosophy-recommender`, promoted via the
`production` alias. The backend loads `philosophy-recommender:production` at
startup — promotion means moving the alias, never changing backend code — and
`/health` reports the resolved version.

## Monitoring definitions

The assignment's monitoring spec assumes a classifier; a recommender requires
explicit translations, stated here and in the dashboard sidebar:

- **Latency** — per-request `latency_ms` from the prediction log, plotted with
  rolling median and p95.
- **Target drift** — the distribution of **recommended authors** over time.
  Sub-genre was the original design, but 94% of catalog items carry no
  category below "Philosophy" (measured), so the category dimension is
  degenerate. Author concentration is arguably the better dimension for this
  domain regardless: philosophy reading is author-anchored, and a model
  collapsing onto a few canonical names is precisely the drift failure worth
  catching.
- **Live accuracy** — the fraction of user feedback events (👍/👎 in the
  frontend, keyed to predictions by `request_id`) marked helpful: a precision
  proxy, always reported alongside its sample size n.

## Testing & CI

- Unit tests (`tests/`) cover the harness contracts: the eligibility
  threshold, the exact hold-out rule including the recency tiebreak, and the
  top-k boundary.
- GitHub Actions (`.github/workflows/ci.yml`) runs ruff + pytest on every PR
  to `main`; branch protection requires the check to pass before merge.
- UI layers were verified behaviorally against the contracts (every widget
  clicked, every write confirmed in DynamoDB, dashboard numbers confirmed to
  move) rather than by line-by-line review — see AI-assisted development below.

## Reproducing / deploying

1. **Data:** run `notebooks/01_data_prep.ipynb` (Colab, CPU) → logs
   `philosophy-dataset` to W&B. Or skip — v2 is already published.
2. **Train:** `python train.py` → two W&B runs + the model artifact. Promote
   the winner by adding the `production` alias in the W&B UI.
3. **AWS:** create DynamoDB tables `predictions` (PK `request_id`) and
   `feedback` (PK `request_id`, SK `asin`), both in us-east-1. Launch two EC2
   instances (Amazon Linux 2023, LabInstanceProfile attached; app instance
   opens ports 22/8000/8501, ops instance 22/8501).
4. **Deploy — app instance:**
   ```bash
   sudo dnf install -y docker git && sudo systemctl start docker
   git clone https://github.com/mallorye/philosophy-book-recommender.git
   cd philosophy-book-recommender
   docker build -f Dockerfile.backend -t backend .
   docker build -f Dockerfile.frontend -t frontend .
   docker run -d --network host -e WANDB_API_KEY=<key> -e AWS_REGION=us-east-1 --name backend backend
   docker run -d --network host -e BACKEND_URL=http://localhost:8000 --name frontend frontend
   ```
5. **Deploy — ops instance:** as above, but build/run `Dockerfile.dashboard`
   (no env vars needed — LabRole covers DynamoDB; the catalog is baked in).

**Example requests:**
```bash
curl http://<APP_EC2_IP>:8000/health
# {"status":"ok","model_version":"philosophy-recommender:v0","catalog_size":736}

curl -X POST http://<APP_EC2_IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"favorites": ["Meditations"], "k": 5}'
# {"request_id":"...","model_version":"philosophy-recommender:v0",
#  "recommendations":[{"asin":"...","title":"...","author":"...","average_rating":4.5},...],
#  "unmatched_favorites":[],"fallback_used":false,"latency_ms":30}
```

**Learner Lab operational notes:** EC2 instances auto-stop at session end and
restart with new public IPs — all screenshots and URLs in the submission were
captured in a single live session. Session credentials rotate per session; the
deployed system avoids them entirely via the IAM instance role.

## AI-assisted development

This project was built with AI assistance (Claude), under an explicit division
of labor:

- **Human-owned — the map:** architecture, component boundaries, data flow,
  the DynamoDB schema, the monitoring metric definitions, dataset scoping and
  pruning thresholds, and model selection. Each of these decisions can be
  explained and defended from memory.
- **Human-owned — the Python contracts:** core logic (the data pipeline
  predicates, the evaluation harness, the similarity/recommendation path) was
  written or line-reviewed by the author; every function's contract is
  understood even where drafting was assisted.
- **AI-implemented — UX and boilerplate:** the Streamlit frontend and
  dashboard implementations were generated from the author's specifications
  (widget contracts, metric definitions, write schemas), then verified
  behaviorally. Dockerfiles, CI YAML, and test scaffolding were AI-drafted.
- **Verification cut both ways:** AI-drafted tests were themselves reviewed;
  one encoded an incorrect expected value and was caught on its first run by
  tracing the hold-out rule against the fixture by hand.

## Future work

Hyperparameter exploration (similarity metric, k, shrinkage); significance
weighting for thin co-rating edges; fuzzy title matching; edition
deduplication at serving time (same author + near-same title); author-level
recommendation rollups; a content-based or hybrid model using themes, schools
of thought, or difficulty — which would capture relations behavior cannot see,
and vice versa; a prediction cache in DynamoDB keyed by the sorted favorite
set, namespaced by model version.