# Real-Time Fraud Detection — MLOps Platform

End-to-end MLOps platform that scores credit-card transactions for fraud in real
time: experiment tracking, a versioned model registry, a CI metric gate,
containerized serving on Kubernetes, drift detection and observability.

See [SPEC.md](SPEC.md) for the full architecture and build plan.

> **Status: in progress — Phase 0 (setup).**
> This README is a placeholder. It gets its real content in Phase 7: the
> architecture diagram, the metrics table, a demo GIF, and the design-decisions
> section. No metrics are reported here yet because no model has been trained yet.

## Why this project reports PR-AUC and not accuracy

The dataset is ~284,807 transactions of which 492 are fraud (**0.17%**). A model
that predicts "never fraud" scores 99.83% accuracy while catching zero fraud, so
accuracy is meaningless here. The primary metric is **PR-AUC**, alongside
**recall at a fixed precision**.

## Development setup

Requires Python 3.12 or 3.13.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip   # Windows
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Dependencies are pinned exactly in `pyproject.toml`; `requirements.lock.txt`
captures the full transitive tree used by Docker and CI.

## Commands

| Purpose | Command |
|---|---|
| Split data | `python -m src.features` |
| Train + log to MLflow | `python -m src.training.train` |
| Serve locally | `uvicorn src.serving.app:app --reload` |
| Full stack | `docker compose up` |
| Deploy to k8s | `helm install fraud ./helm` |

## Licence / data

The dataset is the public, anonymized
[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
set. It contains no PII. It is **not** committed to this repository — see
`src/data/download.py`.
