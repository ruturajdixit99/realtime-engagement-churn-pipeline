"""Defines the churn/disengagement label from REAL observed future
behavior: an active user-week is labeled 1 ("about to disengage") if the
user has zero listening sessions in the following CHURN_HORIZON_WEEKS weeks,
and 0 if they stay active. This is a real behavioral outcome mined from the
data, not a synthetic/assumed label.

Only weeks where (a) the user was actually active that week, and (b) the
full forward horizon is observable within the sampled data (no look-ahead
past the end of the user's data), are kept as labeled examples -- avoiding
label leakage and right-censoring bias.
"""
import numpy as np
import pandas as pd

from src import config


def _future_window_sum_and_availability(sessions: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """For each index t, sum(sessions[t+1 : t+1+h]) and whether all h of
    those future entries actually exist (i.e. t+h < len(sessions))."""
    n = len(sessions)
    future_sum = np.zeros(n)
    available = np.zeros(n, dtype=bool)
    # Suffix sums make this O(n) instead of O(n*h).
    suffix = np.concatenate([np.cumsum(sessions[::-1])[::-1], [0]])  # suffix[t] = sum(sessions[t:])
    for t in range(n):
        end = t + 1 + h
        if end <= n:
            future_sum[t] = suffix[t + 1] - suffix[end]
            available[t] = True
        else:
            future_sum[t] = np.nan
            available[t] = False
    return future_sum, available


def add_churn_labels(weekly_features: pd.DataFrame) -> pd.DataFrame:
    df = weekly_features.sort_values(["user_id", "week"]).reset_index(drop=True).copy()
    h = config.CHURN_HORIZON_WEEKS

    future_sum_col = np.zeros(len(df))
    available_col = np.zeros(len(df), dtype=bool)

    for user_id, idx in df.groupby("user_id").groups.items():
        idx = idx.sort_values()
        sessions = df.loc[idx, "sessions_count"].to_numpy()
        fsum, avail = _future_window_sum_and_availability(sessions, h)
        future_sum_col[idx.to_numpy()] = fsum
        available_col[idx.to_numpy()] = avail

    df["future_sessions_sum"] = future_sum_col
    df["label_disengaging"] = np.where(df["future_sessions_sum"] == 0, 1, 0)

    eligible = (df["sessions_count"] > 0) & available_col
    labeled = df[eligible].copy()
    labeled["label_disengaging"] = labeled["label_disengaging"].astype(int)

    return labeled.drop(columns=["future_sessions_sum"])


if __name__ == "__main__":
    from src.features import build_weekly_features
    from src.ingest import build_sample
    from src.sessionize import add_sessions, session_summary

    df = build_sample()
    sessions = session_summary(add_sessions(df))
    weekly = build_weekly_features(sessions)
    labeled = add_churn_labels(weekly)
    print(f"Labeled active user-weeks: {len(labeled):,}")
    print(f"Disengagement rate: {labeled['label_disengaging'].mean():.4f}")
    print(labeled["label_disengaging"].value_counts())
