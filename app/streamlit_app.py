"""Live-updating dashboard: replays real historical Last.fm listening events
in chronological order and shows engagement/churn-risk scores updating as
events stream in, plus the offline model evaluation for context.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.stream_simulator import get_engine  # noqa: E402

st.set_page_config(page_title="Real-Time Engagement & Churn", layout="wide")
st.title("Real-Time Engagement & Churn Pipeline")
st.caption(
    "Real Last.fm-1K listening history, replayed in true chronological order. "
    "See 'What real-time means here' at the bottom before reading too much into the live pacing."
)

engine = st.cache_resource(get_engine)()

with st.sidebar:
    st.header("Stream controls")
    speed = st.select_slider(
        "Replay speed (historical time compression)",
        options=[10_000, 50_000, 200_000, 1_000_000, 5_000_000],
        value=200_000,
        format_func=lambda v: f"{v:,}x",
    )
    max_delay = st.slider("Max delay between events shown (sec)", 0.05, 3.0, 1.2, 0.05)

    col1, col2, col3 = st.columns(3)
    if col1.button("Start"):
        engine.start(speed_multiplier=speed, max_delay_seconds=max_delay)
    if col2.button("Stop"):
        engine.stop()
    if col3.button("Reset"):
        engine.reset()

    st.divider()
    if config.ARTIFACTS_DIR.joinpath("model_comparison.json").exists():
        with open(config.ARTIFACTS_DIR / "model_comparison.json", encoding="utf-8") as f:
            model_summary = json.load(f)
        st.subheader("Offline model (held-out real users)")
        best = max(model_summary["results"], key=lambda k: model_summary["results"][k]["roc_auc"])
        st.metric("Best model", best.replace("_", " "))
        st.metric("ROC-AUC", model_summary["results"][best]["roc_auc"])
        st.metric("Disengagement rate", f"{model_summary['disengagement_rate']:.1%}")
    else:
        st.warning("No trained model found. Run `python -m src.train` first.")


@st.fragment(run_every="1.5s")
def live_view():
    snap = engine.snapshot()

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Events processed", f"{snap['events_processed']:,} / {snap['total_events']:,}")
    p2.metric("Simulated time", snap["sim_time"] or "—")
    p3.metric("Users seen", len(snap["risk_by_user"]))
    p4.metric("Status", "streaming" if snap["running"] else "stopped")

    if snap["total_events"]:
        st.progress(min(1.0, snap["events_processed"] / snap["total_events"]))

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Live event feed")
        if snap["recent_events"]:
            events_df = pd.DataFrame(snap["recent_events"])
            st.dataframe(
                events_df[["timestamp", "user_id", "artist_name", "track_name", "new_session"]],
                height=420,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Press Start in the sidebar to begin streaming real events.")

    with right:
        st.subheader("Live churn-risk by user")
        risk = snap["risk_by_user"]
        if risk:
            rows = [
                {
                    "user_id": uid,
                    "churn_probability": v["churn_probability"],
                    "sessions_7d": v["sessions_count_7d"],
                    "weeks_since_active": round(v["weeks_since_active"], 2),
                    "last_event": v["last_event_time"],
                }
                for uid, v in risk.items()
                if v["churn_probability"] is not None
            ]
            if rows:
                risk_df = pd.DataFrame(rows).sort_values("churn_probability", ascending=False).head(20)
                st.dataframe(risk_df, height=420, hide_index=True, use_container_width=True)
                st.bar_chart(risk_df.set_index("user_id")["churn_probability"])
            else:
                st.info("Model not loaded — risk scores unavailable. Run `python -m src.train` first.")
        else:
            st.info("No users scored yet.")


live_view()

st.divider()
with st.expander("What 'real-time' means here (and its limits)", expanded=False):
    st.markdown(
        """
**What's real:** every event streamed is a real row from the real Last.fm-1K dataset
(real user IDs, real timestamps, real artists/tracks) — nothing here is generated. The
producer thread replays these events in true chronological order, interleaved exactly as
multiple real users' activity actually occurred. The consumer computes engagement features
and churn-risk scores *incrementally*, from only the events seen so far — it never gets to
peek at "future" events, the same constraint a real production consumer would have.

**What's simulated:**
- **Pacing.** The real dataset spans multiple years; replaying it at real wall-clock speed
  would take years. Historical gaps between events are time-compressed by the speed
  multiplier (capped at a max on-screen delay) so the demo is watchable — this is a
  simulation of live arrival timing, not a claim that these events are happening now.
- **The transport layer.** This uses an in-process Python producer thread + shared state
  (a "Kafka-like queue simulation," as the task called it) rather than a real Kafka broker.
  A production system would add: partitioning by user for horizontal scale, consumer-group
  offset tracking/replay, durability across restarts, and backpressure handling — none of
  which this demo needs to prove the feature-engineering and scoring logic works.
- **Live feature approximation.** The live engine computes rolling features using trailing
  *time windows* (e.g. "sessions in the last 7 days") for O(1)-ish incremental updates,
  which approximates but does not exactly reproduce the calendar-week-bucketed features
  the offline model was trained on (see `src/features.py` vs `src/stream_simulator.py`).
  This is a real, known train/serve skew — documented, not hidden — and is exactly the
  kind of discrepancy a real production ML system has to actively test for.
        """
    )
