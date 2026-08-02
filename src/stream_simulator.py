"""A Kafka-like producer/consumer real-time simulation, built without a real
broker: a producer thread replays REAL historical listening events (from the
sampled Last.fm data) in true chronological order across all sampled users
-- exactly as they'd interleave from a live multi-user event stream -- and a
consumer maintains incrementally-updated rolling engagement state per user,
scoring live churn risk with the offline-trained model as events arrive.

What's real: the events, their order, their relative timing texture, and
the model doing the scoring.
What's simulated: wall-clock pacing (historical gaps are time-compressed so
a 5-year dataset is watchable in minutes -- see README "What real-time
means here") and the fact that this replays historical data rather than a
live production event bus (Spotify's real internal stream isn't available
to us).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import joblib
import pandas as pd

from src import config
from src.ingest import build_sample
from src.train import FEATURE_COLS

SESSION_GAP = pd.Timedelta(minutes=config.SESSION_GAP_MINUTES)
ROLLING_WINDOW = pd.Timedelta(days=7 * config.ROLLING_WEEKS)


@dataclass
class UserState:
    last_event_time: pd.Timestamp | None = None
    last_session_start: pd.Timestamp | None = None
    prior_active_time: pd.Timestamp | None = None  # activity strictly before the current session
    track_events: deque = field(default_factory=deque)  # (timestamp, artist_id, track_id)
    sessions: deque = field(default_factory=deque)  # (start, end)
    day_events: deque = field(default_factory=deque)  # timestamps within trailing 7 days -> "this week"

    def prune(self, now: pd.Timestamp):
        cutoff = now - ROLLING_WINDOW
        while self.track_events and self.track_events[0][0] < cutoff:
            self.track_events.popleft()
        while self.sessions and self.sessions[0][1] < cutoff:
            self.sessions.popleft()
        week_cutoff = now - pd.Timedelta(days=7)
        while self.day_events and self.day_events[0] < week_cutoff:
            self.day_events.popleft()

    def feature_vector(self) -> dict:
        n = len(self.track_events)
        artists = {a for _, a, _ in self.track_events}
        tracks = [t for _, _, t in self.track_events]
        unique_tracks = len(set(tracks))
        repeat_rate = (n - unique_tracks) / n if n else 0.0
        durations = [(e - s).total_seconds() / 60.0 for s, e in self.sessions]
        avg_session_minutes = sum(durations) / len(durations) if durations else 0.0
        weeks_since_active = 0.0
        if self.prior_active_time is not None and self.last_event_time is not None:
            weeks_since_active = max(
                0.0, (self.last_event_time - self.prior_active_time).total_seconds() / (3600 * 24 * 7)
            )
        rolling_sessions = len(self.sessions)
        return {
            "sessions_count": float(len([s for s in self.sessions if s[0] >= (self.last_event_time - pd.Timedelta(days=7))])) if self.last_event_time is not None else 0.0,
            "tracks_count": float(n),
            "unique_artists": float(len(artists)),
            "repeat_rate": float(repeat_rate),
            "avg_session_minutes": float(avg_session_minutes),
            "rolling_sessions": float(rolling_sessions),
            "rolling_tracks": float(n),
            "rolling_unique_artists": float(len(artists)),
            "rolling_repeat_rate": float(repeat_rate),
            "sessions_trend": float(rolling_sessions / config.ROLLING_WEEKS) if rolling_sessions else 0.0,
            "weeks_since_active": float(weeks_since_active),
        }


class LiveEngagementEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._events: pd.DataFrame | None = None
        self._model = None
        self._feature_cols = FEATURE_COLS
        self._user_states: dict[str, UserState] = defaultdict(UserState)
        self._recent_events: deque = deque(maxlen=40)
        self._risk_snapshot: dict[str, dict] = {}
        self._cursor = 0
        self._sim_time: pd.Timestamp | None = None
        self._speed_multiplier = 200_000.0
        self._max_delay_seconds = 1.2

    def _ensure_data_loaded(self):
        if self._events is None:
            df = build_sample()
            self._events = df.sort_values("timestamp").reset_index(drop=True)
        if self._model is None and config.MODEL_PATH.exists():
            bundle = joblib.load(config.MODEL_PATH)
            self._model = bundle["models"]["logistic_regression"]

    def start(self, speed_multiplier: float = 200_000.0, max_delay_seconds: float = 1.2):
        self._ensure_data_loaded()
        with self._lock:
            if self._running:
                return
            self._speed_multiplier = speed_multiplier
            self._max_delay_seconds = max_delay_seconds
            self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False

    def reset(self):
        self.stop()
        time.sleep(0.05)
        with self._lock:
            self._user_states = defaultdict(UserState)
            self._recent_events = deque(maxlen=40)
            self._risk_snapshot = {}
            self._cursor = 0
            self._sim_time = None

    def _process_event(self, row) -> None:
        user_id = row.user_id
        t = row.timestamp
        state = self._user_states[user_id]

        is_new_session = state.last_event_time is None or (t - state.last_event_time) > SESSION_GAP
        if is_new_session:
            if state.last_session_start is not None:
                state.prior_active_time = state.last_event_time
            state.last_session_start = t
            state.sessions.append([t, t])
        else:
            state.sessions[-1][1] = t

        state.track_events.append((t, row.artist_id, row.track_id))
        state.day_events.append(t)
        state.last_event_time = t
        state.prune(t)

        churn_prob = None
        if self._model is not None:
            feats = state.feature_vector()
            X = pd.DataFrame([[feats[c] for c in self._feature_cols]], columns=self._feature_cols)
            churn_prob = float(self._model.predict_proba(X)[0, 1])

        with self._lock:
            self._recent_events.append(
                {
                    "user_id": user_id,
                    "artist_name": row.artist_name,
                    "track_name": row.track_name,
                    "timestamp": t.isoformat(),
                    "new_session": bool(is_new_session),
                }
            )
            self._risk_snapshot[user_id] = {
                "churn_probability": churn_prob,
                "sessions_count_7d": state.feature_vector()["sessions_count"],
                "weeks_since_active": state.feature_vector()["weeks_since_active"],
                "last_event_time": t.isoformat(),
            }
            self._sim_time = t
            self._cursor += 1

    def _run_loop(self):
        events = self._events
        n = len(events)
        prev_t = None
        while self._cursor < n:
            with self._lock:
                if not self._running:
                    return
            row = events.iloc[self._cursor]
            t = row.timestamp
            if prev_t is not None:
                real_gap = (t - prev_t).total_seconds()
                delay = min(max(real_gap / self._speed_multiplier, 0.0), self._max_delay_seconds)
                time.sleep(delay)
            self._process_event(row)
            prev_t = t

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "recent_events": list(self._recent_events)[::-1],
                "risk_by_user": dict(self._risk_snapshot),
                "events_processed": self._cursor,
                "total_events": len(self._events) if self._events is not None else 0,
                "sim_time": self._sim_time.isoformat() if self._sim_time is not None else None,
                "running": self._running,
            }


_engine: LiveEngagementEngine | None = None


def get_engine() -> LiveEngagementEngine:
    global _engine
    if _engine is None:
        _engine = LiveEngagementEngine()
    return _engine
