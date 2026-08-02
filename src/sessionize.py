"""Turns raw real listening events into real sessions.

A "session" is detected the same way music/web analytics literature detects
it from event logs with no explicit session marker: a new session starts
whenever the gap since the user's previous play exceeds SESSION_GAP_MINUTES.
This is standard practice (used in the original Last.fm/MSD session-mining
papers), not a synthetic label -- it's derived purely from real timestamps.
"""
import pandas as pd

from src import config


def add_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"]).copy()
    gap = df.groupby("user_id")["timestamp"].diff()
    new_session = (gap.isna()) | (gap > pd.Timedelta(minutes=config.SESSION_GAP_MINUTES))
    df["session_seq"] = new_session.groupby(df["user_id"]).cumsum()
    df["session_id"] = df["user_id"] + "_" + df["session_seq"].astype(str)
    return df.drop(columns=["session_seq"])


def session_summary(df_with_sessions: pd.DataFrame) -> pd.DataFrame:
    grp = df_with_sessions.groupby(["user_id", "session_id"])
    summary = grp.agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        num_tracks=("track_id", "size"),
        unique_tracks=("track_id", "nunique"),
        unique_artists=("artist_id", "nunique"),
    ).reset_index()
    summary["repeat_tracks"] = summary["num_tracks"] - summary["unique_tracks"]
    summary["duration_minutes"] = (
        summary["end_time"] - summary["start_time"]
    ).dt.total_seconds() / 60.0
    return summary


if __name__ == "__main__":
    from src.ingest import build_sample

    df = build_sample()
    df = add_sessions(df)
    sessions = session_summary(df)
    print(f"Real events: {len(df):,} -> real sessions: {len(sessions):,}")
    print(sessions.select_dtypes(include="number").describe())
