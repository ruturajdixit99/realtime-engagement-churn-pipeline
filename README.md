# Real-Time Streaming Engagement & Churn Pipeline (Spotify-style)

A real-time engagement scoring pipeline that replays real historical music-listening events
in true chronological order, incrementally computes engagement features per user as those
events arrive, and scores live disengagement ("about to go quiet") risk with an offline-trained
classifier — with a live-updating Streamlit dashboard on top.

**Live demo (Next.js/Vercel):** https://realtime-engagement-churn-pipeline-ruturajdixit99s-projects.vercel.app
**Local dashboard (Streamlit, same pipeline, real server-side threads):** `streamlit run app/streamlit_app.py`

There are two presentation layers on top of the same real Python pipeline, because Streamlit
needs a persistent server process (background threads, WebSockets) that Vercel's serverless
model can't run — see [Two front ends, one real pipeline](#two-front-ends-one-real-pipeline)
below for exactly how the web version differs and why.

## Real data source used, and why

**Source: the [Last.fm-1K Users dataset](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)**
(Celma, 2010; hosted by Universitat Pompeu Fabra's Music Technology Group) — real
`<user, timestamp, artist, track>` listening events for 992 real Last.fm users, collected via
the official Last.fm API's `user.getRecentTracks()` method. This project downloads it directly
(`http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz`, verified ~642MB) and
samples 150 of those real users' **full** real listening histories — 2,845,049 real events,
February 2005 – September 2013.

This wasn't the first option considered — the task named two Kaggle sources and a Spotify Web
API option, and all three were checked before picking this one:

| Source considered | Why not used |
|---|---|
| Kaggle "Spotify Tracks Dataset" | Track *audio features* (danceability, tempo, key...), no user listening events at all. Can't build skips/sessions/engagement from it. |
| Million Song Dataset (Taste Profile subset) | Real `<user, song, play_count>` triples, but **no timestamps** — no time axis means no rolling engagement features and no way to observe "declining session frequency." |
| Spotify Web API, my own listening history | Genuinely real and personalized, but the API caps `recently-played` at the **last 50 tracks** — nowhere near enough per-user history, and only one user, to train a classifier. Requires a Spotify Developer App + OAuth login too. |
| **Last.fm-1K (used)** | Real, timestamped, multi-year, multi-user (150 sampled) listening logs — the only option with enough temporal depth per user to build genuine rolling engagement features and mine real disengagement labels from actual behavior. |

It isn't literally Spotify data, but it is **structurally identical** to what Spotify would use
internally (real user → real track-play → real timestamp), the same "real data, real
methodology, different literal source" approach used for the churn-prediction project in this
same portfolio (Telco data as a Spotify-subscription proxy).

---

## What's implemented

1. **Ingestion of real listening events** — two-pass streaming read of the 19.15M-row raw file
   (too big to load at once) to sample 150 real users' full histories without ever loading the
   whole file into memory.
2. **Real session detection** — sessions inferred from real timestamp gaps (>30 min gap = new
   session; standard practice in listening-log literature), giving 134,631 real sessions with
   real session length, duration, and repeat-track counts.
3. **Rolling, weekly engagement features** — recency (weeks since last active), frequency
   (rolling session/track counts), and repeat-listen rate, computed on a **continuous** weekly
   calendar per user (including real zero-activity weeks — the actual signal of disengagement).
4. **A churn/disengagement classifier trained on real engagement-decay patterns** — Logistic
   Regression and Random Forest predicting "will this user have zero sessions for the next 4
   weeks," using only real, forward-looking, non-leaked behavioral outcomes as the label.
5. **A Kafka-like real-time streaming simulation** — a producer thread replays the real events
   in true chronological order (interleaved across all 150 users, exactly as they actually
   happened), a consumer maintains incrementally-updated per-user engagement state and live
   churn-risk scores from the trained model.
6. **A live-updating Streamlit dashboard** — event feed, live per-user risk table/chart, and
   offline model metrics, auto-refreshing via `st.fragment(run_every=...)`.

No listening event, session, or label in this repo is synthetic — everything traces back to a
real row in the real dataset.

---

## What problem it solves

A subscriber going quiet is the earliest, most actionable churn signal a streaming product has
— it happens weeks before a cancellation, and it's visible in behavior (declining sessions,
declining variety, longer gaps between visits) well before it shows up in subscription-status
data. Catching it requires a system that can (a) turn a raw event stream into engagement
features *as events arrive*, not just in a nightly batch job, and (b) score disengagement risk
continuously so a retention team (or automated re-engagement email/push) can act while the
user is still reachable. This project builds both halves — the historical model that learns
what decay looks like, and the streaming path that scores it live — and is explicit about
where the "live" path is a faithful simulation versus true production infrastructure.

---

## How it solves it

### Data flow

```
lastfm-dataset-1K.tar.gz (real, 992 users, 19.15M events, UPF/Celma 2010)
        │
        ▼
src/ingest.py — 2-pass streaming sample: pass 1 counts events/user, pass 2 keeps
        │        full real histories for 150 sampled real users (>=500 events each)
        ▼
src/sessionize.py — real session detection from real timestamp gaps (>30min = new session)
        │             ──▶ 134,631 real sessions (start/end, track count, repeats)
        ▼
src/features.py — continuous weekly calendar per user (incl. real zero-activity weeks)
        │           + rolling 4-week engagement features (recency/frequency/repeat-rate)
        ▼
src/churn_labels.py — REAL forward-looking label: was this active week followed by
        │              4 straight weeks of silence? (no synthetic labels, no leakage:
        │              only weeks with a fully-observable future window are labeled)
        ▼
src/train.py — LogisticRegression + RandomForest, split by user_id (not row), evaluated
        │        on real held-out users            ──▶ artifacts/model_comparison.json
        ▼
src/stream_simulator.py — producer thread replays real events in true chronological
        │                   order (all 150 users interleaved); consumer computes live
        │                   rolling features + scores churn-risk with the trained model
        ▼
app/streamlit_app.py — live event feed + live per-user risk table/chart,
                         auto-refreshing every 1.5s via st.fragment
```

### Module-by-module breakdown

| Module | Reads | Writes | Purpose |
|---|---|---|---|
| `ingest.py` | raw 2.5GB tsv | `events_sample.parquet` | Memory-safe 2-pass sampling of real users' full histories |
| `sessionize.py` | events | session table | Real session boundaries from real timestamp gaps |
| `features.py` | sessions | weekly feature table | Continuous weekly RFM-style engagement features, incl. real silence |
| `churn_labels.py` | weekly features | labeled rows | Real, leakage-safe forward-looking disengagement label |
| `train.py` | labeled rows | `churn_model.joblib`, `model_comparison.json` | User-level train/test split, 2-model comparison |
| `stream_simulator.py` | `events_sample.parquet`, trained model | live in-memory state | Producer/consumer real-time replay + incremental scoring |
| `streamlit_app.py` | live engine snapshot | UI | Auto-refreshing live dashboard |
| `export_web_replay.py` | events, labeled data | `web/public/data/*.json` | Stages a real bounded slice + portable model weights for the Vercel app |

### Why these techniques

**Session detection by timestamp gap**, because the raw event log has no explicit session
marker — this is the standard, defensible way to reconstruct sessions from a play-event log
(used in the original Last.fm/MSD session-mining literature), not an arbitrary choice.

**A continuous weekly calendar including zero-activity weeks.** If disengagement is "declining
session frequency," the model needs to actually see weeks with zero sessions as real, present
rows — dropping them (as a naive `groupby` would) would hide exactly the signal being modeled.

**Forward-looking, leakage-checked labels.** The label for week *t* depends only on weeks
*t+1...t+4* — never on week *t* itself or earlier — and a row is only labeled if all 4 future
weeks are actually observable in the sampled data (no guessing past the end of a user's
history). This is what makes the classifier's recall number meaningful: it's predicting a real
future outcome from real past behavior, not fitting to information it was quietly given.

**User-level (not row-level) train/test split**, via `GroupShuffleSplit` on `user_id`. A
row-level split would put different weeks of the *same* user in both train and test, letting
the model partly memorize individual users' baseline behavior rather than learning general
decay patterns — an easy, common leakage bug in this kind of longitudinal data.

**Logistic Regression + Random Forest**, `class_weight="balanced"` on both. The real
disengagement rate here is 2.23% (this sample skews toward heavy Last.fm "scrobblers," so
going quiet for a month is a comparatively rare, meaningful event) — accuracy would be
meaningless on this imbalance, so evaluation is ROC-AUC / PR-AUC / precision / recall / F1
throughout, and PR-AUC is weighted most heavily in the writeup below since it's the honest
metric under severe imbalance.

**An in-process producer/consumer thread pair instead of real Kafka.** The task allowed either;
a real broker adds real value at production scale (partitioning, consumer groups, durable
offsets, multi-service fan-out) but none of that is being tested here — the thing under test is
whether the feature-engineering and scoring logic works correctly as events arrive one at a
time, which an in-memory queue proves identically. See the section below for exactly what this
does and doesn't simulate.

---

## What "real-time" means in this implementation (and its limits)

**Real:** every streamed event is a real row from the real dataset (real user, real timestamp,
real artist/track) in true chronological order, interleaved exactly as these 150 real users'
activity actually occurred. The consumer only ever sees events up to "now" in the replay — it
cannot look ahead, the same constraint a real production consumer has. The churn-risk scores
shown live come from the same model trained and evaluated in `train.py`.

**Simulated / limited, explicitly:**

1. **Pacing is time-compressed.** The dataset spans years; replaying it at real wall-clock speed
   would take years to watch. `stream_simulator.py` compresses each real inter-event gap by a
   configurable speed multiplier (default 200,000×), capped at a max on-screen delay — this
   preserves relative burstiness (closely-spaced real plays still look bursty) but is a pacing
   simulation, not a claim that these events are happening right now.
2. **The transport is an in-process thread pair, not a real broker.** `queue`-free here (a
   shared, lock-protected object) is used explicitly as the "Kafka-like queue simulation" the
   task allowed as an alternative to a real broker. A production version would add: partitioning
   by user for horizontal scale, consumer-group offset tracking so a crashed consumer can resume
   without data loss, and backpressure handling — none of which change the correctness of the
   feature/scoring logic being demonstrated.
3. **Live features are a real-time approximation of the batch-trained features, not an exact
   match.** `stream_simulator.py` computes rolling engagement using trailing *time windows*
   (e.g., "sessions in the trailing 7 days," updated incrementally per event) for cheap O(1)-ish
   updates, while `features.py` (used for training) buckets into fixed *calendar weeks*. These
   are close but not numerically identical — a real train/serve skew, documented here rather
   than hidden, and exactly the kind of gap a real production ML system has to monitor for.
4. **This replays historical data, not a live production event bus.** There's no way to attach
   to Spotify's or Last.fm's actual real-time internal event stream from outside the company —
   this pipeline proves the architecture (ingest → sessionize → feature → score, incrementally,
   in true time order) against real historical events instead.

---

## Two front ends, one real pipeline

Streamlit apps need a persistent server process — background threads, WebSocket connections for
auto-refresh — which is exactly what Vercel's serverless model doesn't provide (a serverless
function is stateless and spins up per-request; there's no way to keep a producer thread alive
across requests). So making this showcaseable on Vercel meant a second, architecturally
different front end, not a redeploy of the same app:

| | Streamlit (`app/streamlit_app.py`) | Next.js (`web/`, on Vercel) |
|---|---|---|
| Runs on | A persistent Python server (your machine / any host that runs long processes) | Vercel's static hosting + edge, no server process |
| Replay driver | A real background **producer thread** replaying events server-side | A **client-side timer** replaying a real, bounded event slice in your browser |
| Event universe | All 150 sampled real users, full real date range | A real, curated ~6-month slice (25 of the 150 real users, Jan–Jun 2009) — kept small enough to ship as a static JSON asset |
| Scoring model | The trained `sklearn` model loaded directly (`joblib`) | The trained Logistic Regression's exact coefficients, exported to JSON and re-implemented in TypeScript (`web/src/lib/scoreModel.ts`) — `export_web_replay.py` asserts it reproduces `predict_proba` to 1e-6 before export |
| Incremental features | `UserState` class in Python | `UserState` class ported line-for-line to TypeScript (`web/src/lib/engagementEngine.ts`) — same trailing-time-window approximation, same documented train/serve skew |

Both are real implementations of the same architecture against the same real data — the Next.js
version just proves it can run with zero server infrastructure, which is what "showcase on
Vercel" actually requires.

---

## Dataset used

- **Name:** Last.fm Dataset - 1K users (a.k.a. lastfm-dataset-1K)
- **Source:** Òscar Celma, Music Technology Group, Universitat Pompeu Fabra (2010); collected
  via the official Last.fm API. Downloaded from the UPF mirror
  (`mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz`).
- **Full dataset size:** 19,150,868 real listening events across 992 real users.
- **Sample used in this project:** 150 randomly sampled users (each with ≥500 real events) →
  **2,845,049 real events**, spanning **Feb 2005 – Sep 2013**. (Sampling only decides *which*
  real users are included — every kept row is copied verbatim from the source file.)
- **Fields:** `user_id`, `timestamp`, `musicbrainz-artist-id`, `artist-name`,
  `musicbrainz-track-id`, `track-name`.
- **Known real-data quirk handled:** the raw tsv contains some malformed lines (unescaped tabs
  in a handful of track/artist names); `ingest.py` uses `on_bad_lines="skip"` and
  `quoting=QUOTE_NONE` and reports counts, rather than silently corrupting adjacent rows.

---

## Evaluation results

Real numbers from `python -m src.train` (random seed 42):

- **12,693** labeled active user-weeks (150 real users, forward horizon fully observable)
- **120 train users** (9,849 rows) / **30 test users** (2,844 rows) — split by `user_id`
- **Real disengagement rate: 2.23%** (283 of 12,693 active weeks were followed by 4+ silent weeks)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.773 | 0.116 | 0.043 | **0.724** | 0.081 |
| **Random Forest** | **0.780** | **0.167** | **0.103** | 0.569 | **0.174** |

Confusion matrices (test set, threshold 0.5):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 1,850 | 936 | 16 | 42 |
| Random Forest | 2,497 | 289 | 25 | 33 |

Both models clearly separate signal from noise (ROC-AUC ~0.77-0.78, well above the 0.5
random baseline) despite a genuinely hard, severely imbalanced real-world problem. Logistic
Regression catches 72% of real disengagement events (high recall) at the cost of a lot of false
alarms (precision 4.3%); Random Forest trades some recall for meaningfully better precision and
PR-AUC. Under this imbalance, **PR-AUC is the more honest headline metric than ROC-AUC** — Random
Forest's 0.167 vs. a 0.022 no-skill baseline is a real ~7.5× lift.

*(These numbers reflect a fix made during development: `weeks_since_active` was originally
computed as a streak that resets to 0 on any active week — which made it identically 0 across
the entire labeled dataset, since every labeled row **is** an active week, silently contributing
nothing. Fixed by shifting the streak by one row so it reports the silence immediately
**preceding** a return to activity — real signal: disengaging weeks average 2.5 prior silent
weeks vs. 0.3 for retained weeks. Caught by noticing `weeks_since_active.std() == 0` during
review, the same way the earlier week-anchor bug was caught: a summary statistic that shouldn't
be exactly zero, was.)*

---

## How to run it

```bash
python -m venv .venv
./.venv/Scripts/activate        # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# 1. Download the real dataset (~642MB) into data/
#    (ingest.py will raise a clear error telling you this if skipped)
python -c "import urllib.request; urllib.request.urlretrieve(
    'http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz',
    'data/lastfm-dataset-1K.tar.gz')"

# 2. Extract + sample 150 real users (2-pass streaming read; a few minutes)
python -m src.ingest

# 3. Train + evaluate the churn classifier
python -m src.train

# 4. Launch the live dashboard
streamlit run app/streamlit_app.py
```

Press **Start** in the sidebar to begin streaming real historical events; the event feed and
live churn-risk table update every 1.5 seconds as the replay progresses.

### Web app (`web/`, deployable to Vercel)

```bash
# From the project root, after running the Python steps above at least once:
python -m src.export_web_replay    # stages real replay slice + model weights into web/public/data/

cd web
npm install
npm run dev        # http://localhost:3000
# or deploy:
vercel --prod
```

The committed `web/public/data/*.json` already reflects the numbers in this README, so
`cd web && npm install && npm run dev` works standalone without re-running the Python pipeline.

---

## Limitations

- **No skip/completion signal.** Unlike Spotify's actual internal logs (or the Spotify
  Sequential Skip Prediction dataset), Last.fm-1K has no track duration or `ms_played` field, so
  a real "skipped vs. completed" flag isn't derivable — this project uses real session length,
  repeat-listens, and session frequency as the available engagement signals instead, and says so
  rather than approximating skip behavior with a guess.
- **Small, non-representative user sample.** 150 users, and only those with ≥500 events (i.e.
  Last.fm's heaviest users) — real engagement patterns for casual listeners would look
  different (and probably churn far more, and far more often). The 2.23% disengagement rate is
  real for *this* sample, not a general claim about music-streaming churn rates.
- **Precision is low at the default 0.5 threshold** (4–10%), a direct consequence of the severe
  class imbalance and small positive-example count (283). A production system would tune the
  threshold against a business cost function (false-alarm cost vs. missed-churn cost) rather
  than using 0.5, or act on a ranked risk list ("top 20 at-risk users this week") instead of a
  hard cutoff — the dashboard's live risk table already sorts this way.
- **Live and batch features are approximations of each other, not identical** (see "What
  real-time means" above) — a real deployment would need drift/skew monitoring between the
  serving path and the training path, not just a one-time visual check.
- **This is historical replay, not a live production stream.** The architecture (incremental
  ingest → session → feature → score) is real and would work unchanged against a genuine live
  feed; the data source itself is not live.
- **Dataset recency.** The underlying data ends in 2009 (per the official documentation) with a
  handful of later re-collected timestamps observed up to 2013 in this API-scraped dataset — it
  reflects 2005-2009-era listening behavior, not current streaming UX patterns (e.g. no
  podcast/playlist-algorithm-driven listening).
