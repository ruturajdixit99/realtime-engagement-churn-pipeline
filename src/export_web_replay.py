"""Exports a real, bounded slice of the pipeline's data for the Next.js/
Vercel showcase app:

  1. replay_events.json — a real ~6-month window of real events from 25 real
     sampled users (not all 150/2.85M — kept small enough to ship as a
     static JSON asset), sorted chronologically, for the client-side live
     replay animation.
  2. model_weights.json — the trained Logistic Regression's exact scaler
     stats + coefficients, ported to the web app for live scoring, with a
     sanity check that it reproduces sklearn's predict_proba to 1e-6.
  3. engagement_summary.json — real aggregate feature comparisons between
     disengaging vs. retained user-weeks (the FULL labeled dataset, not
     just the replay window) for the dashboard's "why it works" chart.
  4. model_comparison.json — copied from artifacts/ (already real).

No event, timestamp, or number here is synthetic — the only curation is
*which* real users/dates are included in the (necessarily smaller) replay
slice, same principle as the original 150-user sample.
"""
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.churn_labels import add_churn_labels
from src.features import build_weekly_features
from src.ingest import build_sample
from src.sessionize import add_sessions, session_summary
from src.train import FEATURE_COLS

WEB_PUBLIC_DATA_DIR = config.ROOT_DIR / "web" / "public" / "data"

REPLAY_WINDOW_START = pd.Timestamp("2009-01-01", tz="UTC")
REPLAY_WINDOW_END = pd.Timestamp("2009-06-30", tz="UTC")
N_POSITIVE_USERS = 15
N_OTHER_USERS = 10


def select_replay_users(labeled: pd.DataFrame, all_users: list[str]) -> list[str]:
    pos_users = sorted(labeled.loc[labeled["label_disengaging"] == 1, "user_id"].unique())
    rng = np.random.default_rng(config.RANDOM_SEED)
    pos_sample = list(rng.choice(pos_users, size=min(N_POSITIVE_USERS, len(pos_users)), replace=False))
    other_users = sorted(set(all_users) - set(pos_sample))
    other_sample = list(rng.choice(other_users, size=min(N_OTHER_USERS, len(other_users)), replace=False))
    return sorted(pos_sample + other_sample)


def export_replay_events(df: pd.DataFrame, users: list[str]) -> None:
    sub = df[
        df["user_id"].isin(users)
        & (df["timestamp"] >= REPLAY_WINDOW_START)
        & (df["timestamp"] <= REPLAY_WINDOW_END)
    ].sort_values("timestamp")

    records = [
        {
            "t": row.timestamp.isoformat(),
            "u": row.user_id,
            "a": row.artist_name,
            "tr": row.track_name,
        }
        for row in sub.itertuples()
    ]
    out_path = WEB_PUBLIC_DATA_DIR / "replay_events.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Exported {len(records):,} real events ({len(users)} real users, "
          f"{REPLAY_WINDOW_START.date()} to {REPLAY_WINDOW_END.date()}) -> {out_path}")


def export_model_weights(labeled: pd.DataFrame) -> None:
    X = labeled[FEATURE_COLS]
    y = labeled["label_disengaging"]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED)),
        ]
    )
    pipeline.fit(X, y)
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    payload = {
        "feature_cols": FEATURE_COLS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }

    # Sanity check: exported weights must reproduce sklearn's predict_proba.
    X_t = scaler.transform(X.iloc[:10])
    sk_probs = clf.predict_proba(X_t)[:, 1]
    manual_logits = X_t @ np.array(payload["coefficients"]) + payload["intercept"]
    manual_probs = 1 / (1 + np.exp(-manual_logits))
    assert np.allclose(sk_probs, manual_probs, atol=1e-6), "Exported weights do not match sklearn output!"

    out_path = WEB_PUBLIC_DATA_DIR / "model_weights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Exported model weights (sanity check passed) -> {out_path}")


def export_engagement_summary(labeled: pd.DataFrame, df: pd.DataFrame, sessions: pd.DataFrame) -> None:
    grouped = labeled.groupby("label_disengaging")[FEATURE_COLS].mean()
    feature_comparison = [
        {
            "feature": col,
            "retained_avg": round(float(grouped.loc[0, col]), 4),
            "disengaging_avg": round(float(grouped.loc[1, col]), 4),
        }
        for col in FEATURE_COLS
    ]

    summary = {
        "n_users": int(df["user_id"].nunique()),
        "n_events": int(len(df)),
        "n_sessions": int(len(sessions)),
        "n_labeled_user_weeks": int(len(labeled)),
        "disengagement_rate": round(float(labeled["label_disengaging"].mean()), 4),
        "date_range": [df["timestamp"].min().isoformat(), df["timestamp"].max().isoformat()],
        "feature_comparison": feature_comparison,
    }
    out_path = WEB_PUBLIC_DATA_DIR / "engagement_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Exported engagement summary -> {out_path}")


def main():
    WEB_PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = build_sample()
    sessions = session_summary(add_sessions(df))
    weekly = build_weekly_features(sessions)
    labeled = add_churn_labels(weekly)

    users = select_replay_users(labeled, sorted(df["user_id"].unique()))
    export_replay_events(df, users)
    export_model_weights(labeled)
    export_engagement_summary(labeled, df, sessions)

    if (config.ARTIFACTS_DIR / "model_comparison.json").exists():
        shutil.copy(
            config.ARTIFACTS_DIR / "model_comparison.json",
            WEB_PUBLIC_DATA_DIR / "model_comparison.json",
        )
        print("Copied model_comparison.json")


if __name__ == "__main__":
    main()
