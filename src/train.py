"""Trains and compares Logistic Regression and Random Forest classifiers to
predict real user disengagement from real rolling engagement-decay features.
Split by user_id (not by row) so no user's behavior pattern leaks between
train and test.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.churn_labels import add_churn_labels
from src.features import build_weekly_features
from src.ingest import build_sample
from src.sessionize import add_sessions, session_summary

FEATURE_COLS = [
    "sessions_count",
    "tracks_count",
    "unique_artists",
    "repeat_rate",
    "avg_session_minutes",
    "rolling_sessions",
    "rolling_tracks",
    "rolling_unique_artists",
    "rolling_repeat_rate",
    "sessions_trend",
    "weeks_since_active",
]


def build_labeled_dataset() -> "pd.DataFrame":
    df = build_sample()
    sessions = session_summary(add_sessions(df))
    weekly = build_weekly_features(sessions)
    labeled = add_churn_labels(weekly)
    return labeled


def evaluate(y_true, y_prob, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_compare() -> dict:
    labeled = build_labeled_dataset()
    X = labeled[FEATURE_COLS]
    y = labeled["label_disengaging"]
    groups = labeled["user_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE_USERS, random_state=config.RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results[name] = evaluate(y_test, y_prob, y_pred)
        fitted[name] = model

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "feature_cols": FEATURE_COLS}, config.MODEL_PATH)

    summary = {
        "n_labeled_user_weeks": int(len(labeled)),
        "n_users": int(labeled["user_id"].nunique()),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_train_users": int(groups.iloc[train_idx].nunique()),
        "n_test_users": int(groups.iloc[test_idx].nunique()),
        "disengagement_rate": round(float(y.mean()), 4),
        "results": results,
    }
    with open(config.ARTIFACTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    summary = train_and_compare()
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    for name, m in summary["results"].items():
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v}")
