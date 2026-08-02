"""Builds weekly RFM-style rolling engagement features per real user, from
real sessions, over a continuous weekly calendar (including weeks with zero
activity -- those weeks are real signal too: an observed silence).
"""
import numpy as np
import pandas as pd

from src import config


def _monday_of_week(ts: pd.Series) -> pd.Series:
    """Monday 00:00 of the ISO week containing ts, tz-naive. Computed with
    explicit weekday arithmetic (not `.dt.to_period("W-MON")`) so it aligns
    exactly with `pd.date_range(freq="W-MON")` used to build the continuous
    weekly calendar below -- those two use different Monday/Tuesday anchors
    if mixed, which silently drops every real row on merge.
    """
    normalized = ts.dt.tz_localize(None).dt.normalize()
    return normalized - pd.to_timedelta(normalized.dt.weekday, unit="D")


def weekly_activity(sessions: pd.DataFrame) -> pd.DataFrame:
    s = sessions.copy()
    s["week"] = _monday_of_week(s["start_time"])

    weekly = (
        s.groupby(["user_id", "week"])
        .agg(
            sessions_count=("session_id", "nunique"),
            tracks_count=("num_tracks", "sum"),
            unique_artists=("unique_artists", "sum"),
            repeat_tracks=("repeat_tracks", "sum"),
            avg_session_minutes=("duration_minutes", "mean"),
        )
        .reset_index()
    )
    return weekly


def _fill_full_weekly_calendar(weekly: pd.DataFrame) -> pd.DataFrame:
    """Every user gets one row per week from their first to last observed
    week, with zero-activity weeks explicitly present (not missing) -- a
    real user who goes quiet shows up as real zeros, which is exactly the
    engagement-decay signal the churn label is built from.
    """
    frames = []
    for user_id, grp in weekly.groupby("user_id"):
        full_range = pd.date_range(grp["week"].min(), grp["week"].max(), freq="W-MON")
        full = pd.DataFrame({"week": full_range})
        full["user_id"] = user_id
        merged = full.merge(grp, on=["user_id", "week"], how="left")
        for col in ["sessions_count", "tracks_count", "unique_artists", "repeat_tracks"]:
            merged[col] = merged[col].fillna(0)
        merged["avg_session_minutes"] = merged["avg_session_minutes"].fillna(0)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values(["user_id", "week"])


def add_rolling_features(weekly_full: pd.DataFrame) -> pd.DataFrame:
    df = weekly_full.copy()
    df["repeat_rate"] = np.where(
        df["tracks_count"] > 0, df["repeat_tracks"] / df["tracks_count"], 0.0
    )

    grouped = df.groupby("user_id")
    w = config.ROLLING_WEEKS

    df["rolling_sessions"] = grouped["sessions_count"].transform(
        lambda s: s.rolling(w, min_periods=1).sum()
    )
    df["rolling_tracks"] = grouped["tracks_count"].transform(
        lambda s: s.rolling(w, min_periods=1).sum()
    )
    df["rolling_unique_artists"] = grouped["unique_artists"].transform(
        lambda s: s.rolling(w, min_periods=1).mean()
    )
    df["rolling_repeat_rate"] = grouped["repeat_rate"].transform(
        lambda s: s.rolling(w, min_periods=1).mean()
    )

    # Trend: this week's sessions vs the rolling average of the prior window
    # -- a real, simple slope proxy for "engagement is declining right now."
    df["prior_rolling_sessions"] = grouped["rolling_sessions"].shift(1)
    df["sessions_trend"] = df["sessions_count"] - (df["prior_rolling_sessions"] / w)
    df["sessions_trend"] = df["sessions_trend"].fillna(0)

    # Weeks of silence immediately BEFORE this row (recency), computed
    # causally (no look-ahead). A streak-ending-at-row-i value is 0 for any
    # active row by construction, which would make this feature degenerate
    # (always 0) on the labeled dataset -- every labeled row IS an active
    # week. Shifting the streak back by one row fixes this: for an active
    # row, it reports how many consecutive weeks of silence preceded it.
    def silence_streak_ending_at(s: pd.Series) -> pd.Series:
        out = np.zeros(len(s), dtype=int)
        streak = 0
        for i, active in enumerate((s > 0).values):
            streak = 0 if active else streak + 1
            out[i] = streak
        return pd.Series(out, index=s.index)

    df["_silence_streak"] = grouped["sessions_count"].transform(silence_streak_ending_at)
    df["weeks_since_active"] = df.groupby("user_id")["_silence_streak"].shift(1).fillna(0)
    df = df.drop(columns=["_silence_streak"])

    return df.drop(columns=["prior_rolling_sessions"])


def build_weekly_features(sessions: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly_activity(sessions)
    full = _fill_full_weekly_calendar(weekly)
    return add_rolling_features(full)


if __name__ == "__main__":
    from src.ingest import build_sample
    from src.sessionize import add_sessions, session_summary

    df = build_sample()
    sessions = session_summary(add_sessions(df))
    weekly = build_weekly_features(sessions)
    print(weekly.head(10))
    print(f"\n{len(weekly):,} user-weeks across {weekly['user_id'].nunique()} real users")
