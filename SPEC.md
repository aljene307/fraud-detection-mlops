# Real-Time Fraud Detection — MLOps Platform

**Project spec / build plan · Solo, local-first, zero-cost**
Author: Mohamed Aljene · Target: PFE portfolio flagship (finance × cloud/DevOps)

---

## 1. One-line pitch

An end-to-end MLOps platform that scores credit-card transactions for fraud in
real time — with experiment tracking, a versioned model registry, automated
retraining, containerized serving on Kubernetes, drift detection, and full
observability.

**The point is not the model. The point is everything around it.**

---

## 2. What this proves to a recruiter

| Signal | How this project shows it |
|---|---|
| Finance domain | Fraud detection, class imbalance, cost-sensitive thresholds |
| ML engineering | Handling 0.17% positive rate, PR-AUC, threshold tuning |
| MLOps | MLflow tracking + registry, promotion gates, retraining |
| DevOps / CI-CD | GitHub Actions: test → train → evaluate-gate → build |
| Cloud-native | Docker → Helm → Kubernetes (kind/minikube) |
| Observability | Prometheus + Grafana + Evidently drift reports |
| Production thinking | p99 latency, throughput, model versioning, rollback |

If you can talk through *why* each of those pieces exists, you interview like
someone who has run systems, not just trained a notebook model.

---

## 3. Scope — read this twice

**In scope (v1):**
- One dataset, one model family, one serving API.
- Local Kubernetes (no paid cloud required).
- The full operational chain end to end, even if each piece is modest.

**Explicitly OUT of scope (resist the temptation):**
- Multiple competing models / heavy hyperparameter search — a good baseline is enough.
- A fancy front-end — a Grafana dashboard IS the UI.
- Real payment rails, real money, real PII — the data is public and anonymized.
- Distributed Kafka clusters — a single-topic simulation is plenty for v1.

**Rule:** finish the vertical slice first (data → model → API → deploy →
monitor), *then* deepen. A thin end-to-end system beats a deep half-system.

---

## 4. Dataset

**Primary: Credit Card Fraud Detection (ULB / Kaggle).**
- ~284,807 transactions, 492 frauds (≈0.17%) — realistic extreme imbalance.
- Features `V1`–`V28` are PCA-anonymized, plus `Time` and `Amount`, label `Class`.
- Free, well-known, no PII, small enough to run on a laptop.

Why it's good: the imbalance *is* the interesting problem. It forces you to
reason about metrics and thresholds the way real fraud teams do.

**Stretch dataset (project #2 later or v2):** IEEE-CIS Fraud Detection — larger,
richer raw features, more feature-engineering surface.

---

## 5. Architecture

```
                    ┌─────────────────────┐
   raw dataset ───► │  Feature pipeline    │
                    │  (clean / split)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐        ┌──────────────┐
                    │  Training pipeline   │───────►│   MLflow      │
                    │  (XGBoost + eval)    │  logs  │  tracking +   │
                    └──────────┬──────────┘        │  registry     │
                               │ promote if gate    └──────┬───────┘
                               ▼                            │ load "Production"
                    ┌─────────────────────┐                │
   tx simulator ───►│  FastAPI scorer      │◄───────────────┘
   (stream)         │  /predict /metrics   │
                    └──────────┬──────────┘
                               │ Prometheus scrape
                    ┌──────────▼──────────┐        ┌──────────────┐
                    │  Prometheus          │───────►│   Grafana     │
                    └─────────────────────┘        └──────────────┘
                               ▲
                    ┌──────────┴──────────┐
                    │  Evidently drift job │  (incoming vs training dist.)
                    └─────────────────────┘

   All services containerized · deployed via Helm on kind/minikube
   CI/CD: GitHub Actions gates the model before it ships
```

---

## 6. Tech stack (all free / open source)

- **Language/ML:** Python, scikit-learn, XGBoost (or LightGBM), imbalanced-learn
- **Experiment tracking + registry:** MLflow
- **Serving:** FastAPI + Uvicorn
- **Real-time sim:** a Python producer script (upgrade to Kafka/Redpanda in v2)
- **Containers:** Docker, Docker Compose (local dev)
- **Orchestration:** Kubernetes via `kind` or `minikube`, packaged with Helm
- **CI/CD:** GitHub Actions
- **Monitoring:** Prometheus, Grafana
- **Drift:** Evidently AI
- **Testing:** pytest

---

## 7. Repo structure

```
fraud-mlops/
├── CLAUDE.md                  # project context for Claude Code (see §10)
├── README.md                  # the recruiter-facing story + diagram + metrics
├── data/                      # gitignored; download script pulls the CSV
├── src/
│   ├── features/              # cleaning, split, scaling
│   ├── training/              # train.py, evaluate.py, mlflow logging
│   ├── serving/              # FastAPI app, model loader, /predict, /metrics
│   ├── simulator/            # streams transactions to the scorer
│   └── monitoring/           # Evidently drift job
├── tests/                     # pytest: data checks, API contract, metric gate
├── docker/                    # Dockerfiles
├── helm/                      # Helm chart for the k8s deployment
├── .github/workflows/ci.yml   # test → train → gate → build
├── docker-compose.yml         # local full-stack: scorer + mlflow + prom + grafana
└── pyproject.toml
```

---

## 8. Metrics you will showcase (put these in the README)

**Model (imbalance-aware — never report accuracy alone):**
- PR-AUC (primary), ROC-AUC (secondary)
- Recall at a fixed precision (e.g. recall @ 90% precision)
- Confusion matrix at your chosen threshold
- The threshold choice + the false-negative-vs-false-positive cost reasoning

**Operational:**
- p50 / p95 / p99 scoring latency
- Throughput (tx/sec the scorer sustains)
- CI pipeline success rate + train→deploy time
- Drift score over the simulated stream

---

## 9. Build plan (phased — ~2–3 weeks part-time)

**Phase 0 — Setup (½ day)**
Repo, virtualenv, `pyproject.toml`, `CLAUDE.md`, data download script, `.gitignore`.
_Done when:_ `python -m src.features` produces a train/test split.

**Phase 1 — Baseline model + MLflow (1–2 days)**
Train XGBoost with class-imbalance handling (`scale_pos_weight` / SMOTE),
evaluate with PR-AUC + recall@precision, log params/metrics/artifacts to MLflow.
_Done when:_ you can open the MLflow UI and compare 3+ runs.

**Phase 2 — Registry + FastAPI serving + Docker (1–2 days)**
Register the best run, promote it to "Production", build a FastAPI service that
loads the Production model and exposes `/predict` and `/metrics`. Dockerize it.
_Done when:_ `curl` a transaction → get a fraud probability back from a container.

**Phase 3 — Real-time simulation (1 day)**
A producer replays the test set as a stream, hitting `/predict`; the service
logs every prediction (score, latency).
_Done when:_ a running stream shows live scores + latency in logs.

**Phase 4 — CI/CD gate (2 days)**
GitHub Actions: on push → run pytest → train → **fail the build if PR-AUC drops
below a threshold** → build & push the image only if the gate passes.
_Done when:_ a PR with a worse model is blocked automatically.

**Phase 5 — Kubernetes + Helm (2–3 days)**
Spin up `kind`/`minikube`, write a Helm chart, deploy the scorer (+ MLflow) to
the cluster.
_Done when:_ `helm install` brings the service up and it scores through the k8s service.

**Phase 6 — Monitoring + drift (2–3 days)**
Prometheus scrapes `/metrics`; Grafana dashboard for latency/throughput/score
distribution; an Evidently job produces a drift report comparing the live stream
to the training distribution.
_Done when:_ a Grafana dashboard shows live traffic and a drift report renders.

**Phase 7 — Polish (1 day)**
README with the architecture diagram, the metrics table, a 2-minute demo GIF,
and a short "design decisions" section (the interview gold).
_Done when:_ a stranger can understand and run it from the README alone.

---

## 10. `CLAUDE.md` starter (drop this at the repo root)

```md
# Fraud Detection MLOps Platform

## Stack
Python, XGBoost, MLflow, FastAPI, Docker, Kubernetes (kind), Helm,
Prometheus, Grafana, Evidently, GitHub Actions.

## Conventions
- src/ layout, typed functions, pytest for everything testable.
- Never report accuracy for the fraud model — this dataset is 0.17% positive.
  Primary metric is PR-AUC; also track recall at fixed precision.
- Models are loaded ONLY from the MLflow "Production" stage in serving code.

## Commands
- Split data:      python -m src.features
- Train + log:     python -m src.training.train
- Serve locally:   uvicorn src.serving.app:app --reload
- Full stack:      docker compose up
- Deploy to k8s:   helm install fraud ./helm

## Design rules
- The metric gate in CI must block any model below the PR-AUC threshold.
- Keep the vertical slice working at all times; deepen after it's green.
```

---

## 11. Interview talking points (prepare these)

- Why accuracy is meaningless at a 0.17% fraud rate, and what you used instead.
- How you chose the decision threshold (cost of a missed fraud vs a false alarm).
- Why a model registry + promotion stage matters vs shipping a pickle file.
- What your CI metric gate prevents, and why it's better than manual review.
- What drift is, how you detect it, and what you'd automate on top of it.

---

## 12. Definition of done (v1)

A single `docker compose up` (local) **and** a `helm install` (cluster) both
bring up a service that scores a live transaction stream, exposes metrics to a
Grafana dashboard, produces a drift report, and is guarded by a CI pipeline that
refuses to ship a degraded model — all documented in a README a stranger can
follow.

---

*Next project after this: Cloud-Native Fintech Microservices + DevSecOps —
reuses this CI/CD, Kubernetes, and monitoring foundation.*
