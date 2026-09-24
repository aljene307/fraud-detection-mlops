# Real-Time Fraud Detection — MLOps Platform

An end-to-end MLOps platform that scores credit-card transactions for fraud in
real time: validated data ingestion, an XGBoost model tuned for a **0.17%
positive rate**, experiment tracking and a versioned model registry in MLflow, a
FastAPI scoring service, and S3-compatible object storage — all running as a
reproducible Docker Compose stack.

Built as an engineering portfolio project, deliberately **as a working system
rather than a notebook**: training the model is one step among the many that have
to hold for a prediction to reach a caller — and most of the interesting failures
live in the others.

**Phases 0–3 complete** (data → model → registry → serving → load simulation).
CI gate, Kubernetes and monitoring are next — see the [roadmap](#roadmap).

---

## Key results

All figures below were measured on this stack, not estimated.

| | Measured | Why this number |
|---|---|---|
| **PR-AUC** (held-out test) | **0.873** | The honest headline on a 0.17% positive rate |
| **Recall @ 90% precision** | **83.8%** | Catches 5 frauds in 6 while 9 alerts in 10 are real |
| **Throughput**, batch mode | **12,291 tx/s** | 500 transactions per request — the model's ceiling |
| **Server latency p50**, single mode | **4.1 ms** | One transaction per request, as a real caller does |
| **Server latency p50**, batch mode | **7.7 ms** / request of 500 | 0.015 ms per transaction |
| **Tests** | **275** passing | Including one that proves the label never leaves the client |

**Accuracy is never reported, and never even computed.** On this dataset a model
that answers "never fraud" is 99.83% accurate and catches nothing. A runtime
guard rejects any metric key containing `accuracy`, and `mlflow.autolog()` is
banned for logging `training_accuracy_score` behind your back.

For the same reason, note that the flattering number here would be **ROC-AUC
0.979** — the metric most fraud projects lead with. The 0.106 gap between that
and PR-AUC 0.873 is the entire point: ROC-AUC's false-positive rate is diluted
by 284,315 true negatives, so it barely moves when the model floods analysts
with false alerts.

---

## Architecture

```
 ┌─ data ─────────────┐     ┌─ validation ───────┐     ┌─ model ────────────┐
 │ Kaggle ULB, public │     │ SHA-256 pinned     │     │ XGBoost            │
 │ 284,807 tx         │ ──▶ │ row + column count │ ──▶ │ scale_pos_weight   │
 │ 492 fraud (0.17%)  │     │ fails LOUDLY, and  │     │ vs SMOTE vs linear │
 │ not in this repo   │     │ never silently     │     │ PR-AUC on test     │
 └────────────────────┘     └────────────────────┘     └──────────┬─────────┘
                                                                  │
                      params · metrics · artifacts · git SHA      │
              ┌───────────────────────────────────────────────────┘
              │
 ╔════════════╪════ docker compose ═════════════════════════════════════════════╗
 ║            ▼                                                                 ║
 ║  ┌─ MLflow :5000 ─────┐     ┌─ registry ─────────┐     ┌─ MinIO :9000 ────┐  ║
 ║  │ tracking server    │ ──▶ │ fraud-detector     │     │ S3-compatible    │  ║
 ║  │ SQLite backend     │     │ @production ALIAS  │ ◀─▶ │ object storage   │  ║
 ║  │ (MLflow 3 refuses  │     │ (stages deprecated │     │ artifacts live   │  ║
 ║  │  the file store)   │     │  since 2.9)        │     │ here as s3://    │  ║
 ║  └────────────────────┘     └──────────┬─────────┘     └──────────────────┘  ║
 ║                                        │                                     ║
 ║                       models:/fraud-detector@production                      ║
 ║                                        │                                     ║
 ║                                        ▼                                     ║
 ║  ┌─ FastAPI scorer :8000 ──────────────────────────┐   ┌─ simulator ──────┐  ║
 ║  │ POST /predict        POST /predict/batch        │   │ replays the test │  ║
 ║  │ GET  /health  /ready  /docs  /metrics           │◀──│ set as a stream, │  ║
 ║  │ Pydantic: 30 fields, extra="forbid", no NaN     │   │ open-loop clock  │  ║
 ║  │ feature order read from the DEPLOYED signature  │   │ profiles: [sim]  │  ║
 ║  └──────────────────────┬──────────────────────────┘   └──────────────────┘  ║
 ║                         │ Prometheus exposition                              ║
 ╚═════════════════════════╪════════════════════════════════════════════════════╝
                           ▼
              ┌─ Prometheus + Grafana ──┐
              │ latency histograms,     │  ← Phase 6
              │ throughput, score dist. │
              └─────────────────────────┘
```

Three properties of this design are worth more than the diagram:

**Serving never names a model file.** It resolves
`models:/fraud-detector@production` — never a run ID, never a pickle. Rolling
back a bad model is repointing an alias: no rebuild, no redeploy, no code
change. Promotion is never automatic; the alias only moves on an explicit
`--promote` that clears a PR-AUC floor.

**The API contract comes from the deployed model, not from the training code.**
Feature order is read out of the MLflow model *signature*, so a model trained
with different columns cannot be silently fed the wrong ones.

**Artifacts live in object storage, not a shared folder.** A file-based artifact
store writes absolute paths into the MLflow database — `file:C:/Users/.../mlruns/...`
— which do not exist inside a Linux container, and no volume mount fixes it
because the *database* dictates the path. `s3://` says **what**,
`MLFLOW_S3_ENDPOINT_URL` says **where**. The same model then loads from Windows,
from a container, and tomorrow from Kubernetes with nothing in the database
changing. MinIO swaps for real S3, GCS or Azure Blob by changing one endpoint.

---

## Stack

| Layer | Tooling |
|---|---|
| Model | **XGBoost**, scikit-learn, imbalanced-learn (SMOTE) |
| Tracking & registry | **MLflow 3** (SQLite backend, `@production` alias) |
| Serving | **FastAPI** + uvicorn, **Pydantic v2** schemas |
| Packaging | **Docker**, Docker Compose, one shared image |
| Object storage | **MinIO** (S3-compatible), boto3 |
| Observability | **prometheus-client** — counters, histograms, model-info gauge |
| Testing | **pytest**, 275 tests, ruff |
| Next | GitHub Actions, Kubernetes + Helm, Grafana, Evidently |

---

## Quick start

```bash
git clone https://github.com/aljene307/fraud-detection-mlops.git
cd fraud-detection-mlops
cp .env.example .env            # dev-only credentials, git-ignored
docker compose up -d --build    # first build takes a few minutes
```

Four services come up: **MinIO** (console on :9001), a one-shot bucket creator,
the **MLflow** tracking server and registry (:5000), and the **scorer** (:8000).

### Endpoints

| | |
|---|---|
| `GET /docs` | Interactive OpenAPI UI — try a real fraud in the browser |
| `POST /predict` | One transaction → probability, decision, threshold, model version |
| `POST /predict/batch` | Up to 1,000 transactions in one call |
| `GET /health` | Liveness. Answers **200 even with no model loaded** |
| `GET /ready` | Readiness. **503 with a reason** until a model is live |
| `GET /metrics` | Prometheus exposition |

`/health` and `/ready` answering differently is deliberate, and it is the single
most important behaviour in the service. A liveness probe that fails because the
model registry is empty gets the container killed and restarted forever, turning
a recoverable problem into a crash loop. So `/health` reports *the process is
alive*, `/ready` reports *I can actually score*, and a load balancer reads the
second one.

That is why a **fresh** stack starts degraded: the registry is empty, so `/ready`
answers 503 and says so. Populate it:

```powershell
# PowerShell. Compose reads .env by itself; your shell does not, so set the
# same two values here. The host talks to localhost, containers talk to names.
$env:MLFLOW_TRACKING_URI    = "http://localhost:5000"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID      = "<MINIO_ROOT_USER from your .env>"
$env:AWS_SECRET_ACCESS_KEY  = "<MINIO_ROOT_PASSWORD from your .env>"

.venv\Scripts\python.exe -m src.data.download   # ~144 MB, SHA-256 verified
.venv\Scripts\python.exe -m src.features        # 60/20/20 split
.venv\Scripts\python.exe -m src.training.train --promote

docker compose restart scorer
curl http://localhost:8000/ready
```

The download step needs a Kaggle API token at `~/.kaggle/kaggle.json`; the
script prints the exact steps if it is missing.

### Replaying the test set as a stream

```bash
# Model throughput ceiling: 500 transactions per request
docker compose run --rm simulator --mode batch --batch-size 500 --transactions 20000

# What a real caller experiences: one transaction per request, at a held rate
docker compose run --rm simulator --mode single --rate 120 --duration 30
```

The simulator prints a live view — current throughput, last score, rolling
latency — then a report putting **client-side and server-side latency side by
side**. Their difference is everything that is not the model.

It sits behind `profiles: [sim]` so `docker compose up` never starts it: it
*writes to the metrics it measures*, and started with the stack every
`compose up` would inject 20,000 transactions into the very counters the Phase 6
dashboards are meant to observe. Naming it in `docker compose run` activates its
profile automatically.

---

## Roadmap

| Phase | | |
|---|---|---|
| **0** | ✅ | Repo, packaging, dataset download with hard validation, 60/20/20 split |
| **1** | ✅ | Three imbalance strategies, PR-AUC + recall@precision, MLflow tracking, registry + `@production` alias, versioned quality gate |
| **2** | ✅ | FastAPI scorer with degraded start, Pydantic contract, Prometheus metrics, Dockerfile + Compose stack on MinIO |
| **3** | ✅ | Stream simulator with open-loop scheduling, client vs server latency, as a profiled Compose service |
| **4** | ⬜ | **CI/CD gate** — GitHub Actions runs pytest, retrains, and fails the build if PR-AUC drops below `config/gate.yaml`; the image is built only if the gate passes |
| **5** | ⬜ | **Kubernetes + Helm** — chart for the scorer and MLflow, deployed on a local cluster |
| **6** | ⬜ | **Monitoring + drift** — Prometheus scraping, Grafana dashboards, an Evidently drift report against the training distribution |
| **7** | ⬜ | Polish — demo GIF, expanded design-decision notes |

Each phase is a separate commit with the reasoning in its message; `SPEC.md`
holds the full plan and `NOTES.md` the running engineering log.

---
---

# Design decisions

The rest of this file is the engineering detail behind the summary above:
why each choice was made, what the alternative cost, and what was measured
rather than assumed.

## Why PR-AUC, concretely

The dataset is 284,807 transactions of which 492 are fraud (**0.17%**). Two
metrics that look interchangeable are not:

- **ROC-AUC** plots recall against the false-positive rate. That rate has 284,315
  true negatives in its denominator, so a thousand extra false alarms move it by
  0.0035. It reports 0.979 here.
- **PR-AUC** plots precision against recall. Precision's denominator is *the
  alerts you raised*, which is small, so the same thousand false alarms wreck it.
  It reports 0.873.

The second is the one that tracks whether a fraud analyst's queue is usable, so
it is the primary metric, alongside **recall at a fixed precision**. PR-AUC is
computed with `average_precision_score`, not `auc(recall, precision)`: the
trapezoid rule interpolates between operating points that no threshold actually
produces.

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

Leakage is prevented structurally rather than by discipline: stateless
transforms (`log_amount`, `hour_of_day`) run before the split, the
`StandardScaler` lives **inside** the sklearn Pipeline so it is fitted per fold,
and `compute_scale_pos_weight()` accepts only `y_train` — there is no argument
you could pass it that would leak.

A `split_manifest.json` records the seed, strategy, per-fold counts and the
source file's SHA-256, so any split is reproducible and traceable.

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

## Load measurement: what the simulator gets right

Both throughput numbers are legitimate and they measure different things. A batch
request pays the per-request cost (~0.2 ms of validation and serialisation) once
for 500 transactions, so it measures **the model's** capacity. A real fraud API
is called one transaction at a time at authorisation, so `--mode single` is the
latency a caller actually sees.

**Where you run the load generator from dominates the result.** Measured on this
stack: `POST /predict` costs **54.81 ms** from Windows against `localhost:8000`
but **6.12 ms** from inside the Docker network — 9× pessimistic, and the
contamination is Docker Desktop's port forwarding, not the service. The trap is
sneaky because `GET /health` is 1.57 ms versus 1.33 ms, near-identical, so a
naive smoke test finds nothing wrong. Hence the simulator is a Compose service
on the internal network; run from the host it prints an explicit warning.

**Sends are scheduled on the clock, not chained to responses.** A closed-loop
generator that waits for each response before sending the next one
mechanically slows down when the server slows down, so the queueing a real
traffic pattern would suffer never appears and high percentiles come out
flattering — *coordinated omission*. Latency is therefore measured from the
*planned* instant, and the resulting scheduling delay is reported rather than
hidden. Not a cosmetic difference: a measured run showed a server p99 of
**9.69 ms** against a client p99 of **210.87 ms**.

Also handled: warm-up requests are sent but excluded, percentiles are suppressed
below 100 samples, one `requests.Session` is reused so TCP handshakes are not
timed, and `time.perf_counter` is used because `time.time()` is subject to NTP
adjustments.

`Class` is separated from the payloads and never sent — a real caller does not
have it. It is kept aside to report observed recall, which the report labels as
simulation-only. A test serialises every outgoing request and greps it to prove
the label never escapes.

## Regenerating the dependency lock

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

## Development setup

Requires Python 3.12 or 3.13.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip   # Windows
.venv/Scripts/python.exe -m pip install -e ".[dev]"
python -m pytest                                        # 275 tests
```

Dependencies are pinned exactly in `pyproject.toml`; `requirements.lock.txt`
captures the full transitive tree used by Docker and CI.

### Commands

| Purpose | Command |
|---|---|
| Download + validate the dataset | `python -m src.data.download` |
| Split data | `python -m src.features` |
| Train + log to MLflow | `python -m src.training.train` |
| Promote the best run | `python -m src.training.train --promote --min-pr-auc 0.80` |
| Browse experiments | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Apply the quality gate | `python -m src.training.gate --report reports/training/xgb.json` |
| Serve locally | `uvicorn src.serving.app:app --reload` |
| Full stack | `docker compose up -d --build` |
| Replay a stream | `docker compose run --rm simulator --mode batch` |
| Regenerate the lock | `python scripts/freeze_lock.py` |

## Licence / data

The dataset is the public, anonymized
[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
set (see that page for its licence terms). It contains no PII: features `V1`–`V28`
are PCA components, and only `Time` and `Amount` are original. It is **not**
committed to this repository — `python -m src.data.download` fetches it and
verifies its SHA-256.

Credentials in `.env.example` are **dev-only placeholder values for a local
MinIO container**. The real `.env` is git-ignored and has never been committed.
