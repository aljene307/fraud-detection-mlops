# Fraud Detection MLOps Platform

See SPEC.md for the full project spec and build plan.

## Stack
Python, XGBoost, MLflow, FastAPI, Docker, Kubernetes (kind), Helm,
Prometheus, Grafana, Evidently, GitHub Actions.

## Conventions
- src/ layout, typed functions, pytest for everything testable.
- Never report accuracy for the fraud model — this dataset is 0.17% positive.
  Primary metric is PR-AUC; also track recall at fixed precision.
- Models are loaded ONLY via the MLflow alias `models:/fraud-detector@production`
  in serving code — never a run URI, never a pickle. (Registry *stages* are
  deprecated since MLflow 2.9; aliases replace them. See src/config.py.)

## Commands
- Split data:      python -m src.features
- Train + log:     python -m src.training.train
- Promote best:    python -m src.training.train --promote --min-pr-auc 0.80
- Check the gate:  python -m src.training.gate --report reports/training/xgb.json
                   (exit 0/1; floor lives in config/gate.yaml, versioned)
- MLflow UI:       mlflow ui --backend-store-uri sqlite:///mlflow.db
                   (SQLite, NOT ./mlruns — MLflow 3 refuses the file store,
                   and the model registry has never worked with it)
- Serve locally:   uvicorn src.serving.app:app --reload
- Full stack:      cp .env.example .env && docker compose up -d --build
                   (minio + mlflow + scorer; see README for the migration steps)
- Regen the lock:  python scripts/freeze_lock.py
                   (NOT a bare `pip freeze` — a Windows lock carries pywin32,
                   which does not exist on Linux and breaks the Docker build)
- Deploy to k8s:   helm install fraud ./helm

## Design rules
- The metric gate in CI must block any model below the PR-AUC threshold.
- Keep the vertical slice working at all times; deepen after it's green.


## How I want you to work with me (IMPORTANT)
- I am learning. Do NOT dump large amounts of code silently.
- Work one small step at a time. Before writing code, explain in plain
  language WHAT you'll do and WHY, then wait for my "go".
- After writing code, walk me through it: what each part does, why this
  approach over alternatives, and what would break without it.
- When you introduce a new concept (PR-AUC, SMOTE, model registry, Helm
  chart, etc.), give me a 2-3 sentence explanation before using it.
- Prefer teaching me the reasoning over saving time. I care about
  understanding, not speed.
