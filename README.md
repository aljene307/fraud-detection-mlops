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
| Apply the quality gate | `python -m src.training.gate --report reports/training/xgb.json` |
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

## Running the stack in Docker

```bash
cp .env.example .env          # dev-only credentials, git-ignored
docker compose up -d --build  # first build pulls ~1.5 GB, takes a few minutes
```

Four services come up: **MinIO** (S3-compatible object storage, console on
:9001), a one-shot **bucket creator**, the **MLflow** tracking server and
registry (:5000), and the **scorer** (:8000). Inside the Docker network services
reach each other by name (`http://minio:9000`); from the host it is `localhost`
and the published port.

On a fresh stack the registry is empty, so the scorer starts **degraded** —
`/health` answers 200, `/ready` answers 503 with the reason. That is the
designed behaviour, not a failure. Populate the registry:

```bash
# PowerShell — the host talks to localhost, containers talk to service names
$env:MLFLOW_TRACKING_URI    = "http://localhost:5000"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID      = "fraud-dev"
$env:AWS_SECRET_ACCESS_KEY  = "fraud-dev-password"

.venv\Scripts\python.exe -m src.training.train --promote
docker compose restart scorer
curl http://localhost:8000/ready
```

### Replaying the test set as a stream

```bash
# Model throughput ceiling: one request carries 500 transactions
docker compose run --rm simulator --mode batch --batch-size 500 --transactions 20000

# What a real caller experiences: one transaction per request, at a held rate
docker compose run --rm simulator --mode single --rate 120 --duration 30
```

The simulator sits behind `profiles: [sim]`, so `docker compose up` never starts
it. Naming it in `docker compose run` activates its profile automatically — no
`--profile sim` needed. It must stay out of the default stack because it *writes
to the metrics it measures*: started with the stack, every `compose up` would
inject transactions into the very counters the Phase 6 dashboards observe.

Both numbers are legitimate and they measure different things. A batch request
pays the per-request cost (~0.2 ms of validation and serialisation) once for 500
transactions, so it measures the **model's** capacity. A real fraud API is
called one transaction at a time at authorisation, so `--mode single` is the
latency a caller actually sees.

The report puts client-side and server-side latency side by side; their
difference is everything that is not the model. Under a held rate it also shows
the **scheduling delay** — sends are planned on the clock rather than chained to
responses, so a stall shows up instead of silently throttling the load
generator. That difference is not cosmetic: a measured run showed a server p99
of 9.69 ms against a client p99 of 210.87 ms.

`Class` is separated from the payloads and never sent — a real caller does not
have it. It is kept aside to report observed recall, which the report labels as
simulation-only.

### Why object storage rather than a shared folder

A file-based artifact store writes **absolute paths** into the MLflow database:

```
file:C:/Users/.../mlruns/1/9add.../artifacts     what AND where, inseparably
s3://mlflow/1/9add.../artifacts                  what
MLFLOW_S3_ENDPOINT_URL=http://minio:9000         where — configuration
```

Those absolute paths do not exist inside a Linux container, and no volume mount
fixes that because the database dictates the path. Object storage splits the two
concerns: the URI names the object, an environment variable says where to ask.
The same model then loads from Windows, from a container, and from Kubernetes
with nothing in the database changing.

MinIO is swapped for real S3, GCS or Azure Blob by changing one endpoint.

### Regenerating the dependency lock

Use `python scripts/freeze_lock.py`, not a bare `pip freeze`, which gets it
wrong twice and both failures are silent.

It would carry the **dev group**: the local venv is installed with
`pip install -e ".[dev]"`, so `pip freeze` sees pytest, ruff, coverage and
httpx2 — and the image installs from this lock, so it would ship the test
tooling to production. The script resolves the runtime closure and keeps only
those names (12 packages dropped).

It would also carry **`pywin32`**, pulled in transitively by mlflow on Windows
and absent from Linux, failing the Docker build with
`No matching distribution found`. The script adds the PEP 508 marker
`sys_platform == "win32"`, which keeps one lock file valid on both platforms.

Versions come from what is *installed*, not from a fresh resolution: a fresh
resolve proposed newer versions for 27 packages. A lock should pin what was
tested, not what happens to be available today.

## The quality gate

`config/gate.yaml` holds the PR-AUC floor CI enforces, and it is versioned:
moving it is a deliberate act visible in review, not a setting that drifts.
`python -m src.training.gate` reads it and exits 0 or 1, so Phase 4's CI step is
one line with no log parsing.

The floor is **0.80**, and the file records why. The same XGBoost model scored
0.8234 on validation and 0.8727 on test — a 0.049 swing caused by nothing but
which frauds landed in which fold, since each holds only ~98 of them. That is an
empirical measurement of fold noise, and it brackets the floor from both sides:
it must sit below the worst fold of a good model (0.8234) or CI would reject the
model we want to keep, and above the linear baseline (0.7602) or a model that
lost everything XGBoost contributes would sail through. The usable window is
[0.78–0.81].

The gate answers "is this model still fit to ship?", not "is this the best we
have ever had?". Catching a slide from 0.8727 to 0.84 needs a comparison against
the current `@production` model rather than a fixed floor — a refinement for
after Phase 4, once the floor's stability has been observed.

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
