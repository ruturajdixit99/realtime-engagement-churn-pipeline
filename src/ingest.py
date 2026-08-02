"""Extracts the real Last.fm-1K tarball and pulls a tractable, real random
sample of users' FULL listening histories into a parquet file.

Two-pass streaming approach (the raw tsv is ~2.5GB / 19.15M rows, too big
to load into memory at once):
  Pass 1: stream the file in chunks, counting events per user_id only.
  Pass 2: stream again, keeping only rows for the sampled user_ids.

No event is fabricated or altered — every row kept is copied verbatim from
the real dataset; the only thing "sampled" is which real users are included.
"""
from __future__ import annotations

import csv
import tarfile

import pandas as pd

from src import config

COLUMNS = ["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"]


def ensure_extracted() -> None:
    if (config.EXTRACT_DIR / config.EVENTS_TSV_NAME).exists():
        return
    if not config.TARBALL_PATH.exists():
        raise FileNotFoundError(
            f"{config.TARBALL_PATH} not found. Download it from {config.DOWNLOAD_URL} first."
        )

    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {config.TARBALL_PATH.name} (this can take a few minutes)...")
    with tarfile.open(config.TARBALL_PATH, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(config.EVENTS_TSV_NAME) or member.name.endswith(
                config.PROFILE_TSV_NAME
            ):
                member.name = member.name.split("/")[-1]  # flatten path
                tar.extract(member, path=config.EXTRACT_DIR)
    print(f"Extracted to {config.EXTRACT_DIR}")


def _events_path():
    return config.EXTRACT_DIR / config.EVENTS_TSV_NAME


def count_events_per_user(chunksize: int = 500_000) -> pd.Series:
    counts: dict[str, int] = {}
    reader = pd.read_csv(
        _events_path(),
        sep="\t",
        header=None,
        names=COLUMNS,
        usecols=["user_id"],
        chunksize=chunksize,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    for i, chunk in enumerate(reader):
        vc = chunk["user_id"].value_counts()
        for uid, c in vc.items():
            counts[uid] = counts.get(uid, 0) + int(c)
        print(f"  pass 1: processed chunk {i + 1} ({(i + 1) * chunksize:,} rows scanned)")
    return pd.Series(counts).sort_values(ascending=False)


def sample_users(user_counts: pd.Series) -> list[str]:
    eligible = user_counts[user_counts >= config.MIN_EVENTS_PER_USER]
    n = min(config.N_SAMPLE_USERS, len(eligible))
    sampled = eligible.sample(n=n, random_state=config.RANDOM_SEED)
    print(f"Sampled {n} real users out of {len(eligible)} eligible (>= {config.MIN_EVENTS_PER_USER} events)")
    return sorted(sampled.index.tolist())


def load_sampled_events(user_ids: list[str], chunksize: int = 500_000) -> pd.DataFrame:
    user_set = set(user_ids)
    parts = []
    reader = pd.read_csv(
        _events_path(),
        sep="\t",
        header=None,
        names=COLUMNS,
        chunksize=chunksize,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        on_bad_lines="skip",
        parse_dates=["timestamp"],
    )
    for i, chunk in enumerate(reader):
        keep = chunk[chunk["user_id"].isin(user_set)]
        if not keep.empty:
            parts.append(keep)
        print(f"  pass 2: processed chunk {i + 1}, kept {sum(len(p) for p in parts):,} rows so far")
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def build_sample(force: bool = False) -> pd.DataFrame:
    if config.EVENTS_PARQUET_PATH.exists() and not force:
        return pd.read_parquet(config.EVENTS_PARQUET_PATH)

    ensure_extracted()
    print("Pass 1/2: counting real events per user...")
    counts = count_events_per_user()
    user_ids = sample_users(counts)
    print("Pass 2/2: loading full histories for sampled real users...")
    df = load_sampled_events(user_ids)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.EVENTS_PARQUET_PATH, index=False)
    print(f"Saved {len(df):,} real events for {df['user_id'].nunique()} real users to {config.EVENTS_PARQUET_PATH}")
    return df


if __name__ == "__main__":
    df = build_sample(force=True)
    print(df.head())
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Events per user (describe):\n{df.groupby('user_id').size().describe()}")
