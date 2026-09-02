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

## How the data is split

`python -m src.features` writes a **3-way 60/20/20** split to `data/processed/`.
Two strategies are available.

**`--strategy=stratified` (default).** A random split that draws separately from
the fraud and non-fraud pools, so all three folds keep the population fraud rate
of ~0.17%. With only 492 positives in the whole dataset, an unstratified draw
would hand one fold 80 frauds and another 115 purely by chance, and validation
metrics would swing for reasons unrelated to the model.

**`--strategy=time`.** Sorts by `Time` and cuts chronologically: train on the
past, test on the future. This mirrors what actually happens in production and
is the more honest estimate of deployed performance. The trade-off is that
stratification becomes impossible by construction — you cannot both fix the
class proportions and respect chronology — so each fold's fraud rate is
inherited rather than chosen, and with so few positives the test fold's metrics
get noisy.

| | stratified | time |
|---|---|---|
| Fraud rate per fold | controlled (~0.17%) | inherited, uneven |
| Metric variance | low | higher |
| Realism | ignores time ordering | matches deployment |
| Used for | the CI gate baseline | a sanity check on optimism |

The default is `stratified` because the Phase 4 CI gate needs a low-variance,
reproducible number: a gate that trips on split noise is a gate nobody trusts.
The time-based split is kept as a flag to quantify how optimistic the stratified
number is.

**The test fold is never used to make decisions.** The decision threshold and
all model comparisons are settled on the validation fold; test is read once, for
the final reported figure. Choosing a threshold on test and then reporting a
metric on that same test inflates the result.

A `split_manifest.json` records the seed, strategy, per-fold counts and the
source file's SHA-256, so any split is reproducible and traceable.

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
| Promote the best run | `python -m src.training.train --promote --min-pr-auc 0.80` |
| Browse experiments | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Serve locally | `uvicorn src.serving.app:app --reload` |
| Full stack | `docker compose up` |
| Deploy to k8s | `helm install fraud ./helm` |

## Experiment tracking and the model registry

Tracking uses a **SQLite** backend (`sqlite:///mlflow.db`), not the classic
`./mlruns` directory: MLflow 3 refuses the filesystem store outright, and the
model registry — which the `@production` alias depends on — has never worked
with it. Override with the `MLFLOW_TRACKING_URI` environment variable to point
at a remote server; no code changes.

Every run records its parameters, `val_`/`test_`-prefixed metrics, a PR curve
and confusion matrix, and tags carrying the git commit, **whether the working
tree was dirty**, and the dataset's SHA-256. A run made from a dirty tree is not
reproducible from its commit alone, so that fact is recorded rather than hidden.

`mlflow.autolog()` is deliberately **not** used: it logs
`training_accuracy_score` automatically, which would break this project's rule
against reporting accuracy. Logging is explicit, and a runtime guard rejects any
metric key containing `accuracy`.

**Promotion is never automatic.** Training registers every run as a new version,
but the `@production` alias only moves when you pass `--promote`, and only if the
best run clears `--min-pr-auc`. A run that finishes should not become production
merely because it finished; in Phase 4 the CI gate takes over that authority.
Serving resolves `models:/fraud-detector@production`, so rolling back is
repointing the alias — no redeploy, no code change.

## Licence / data

The dataset is the public, anonymized
[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
set. It contains no PII. It is **not** committed to this repository — see
`src/data/download.py`.
