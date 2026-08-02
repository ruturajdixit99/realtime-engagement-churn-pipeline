"""Central configuration for the real-time engagement/churn pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

TARBALL_PATH = DATA_DIR / "lastfm-dataset-1K.tar.gz"
EXTRACT_DIR = DATA_DIR / "lastfm-dataset-1K"
EVENTS_TSV_NAME = "userid-timestamp-artid-artname-traid-traname.tsv"
PROFILE_TSV_NAME = "userid-profile.tsv"

# The real UPF Music Technology Group mirror of Celma's 2010 Last.fm-1K
# dataset: <user, timestamp, artist, track> tuples from 992 real Last.fm
# users' full listening history (through May 2009), collected via the
# official Last.fm API's user.getRecentTracks() method.
DOWNLOAD_URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"

EVENTS_PARQUET_PATH = DATA_DIR / "events_sample.parquet"

# Full dataset is 19.15M events / 992 users. For a tractable local demo
# (fast rebuilds, a real-time replay that finishes in minutes not hours)
# this pipeline works off a real random sample of users rather than all
# 992 -- every event used is still a real, unmodified row from the dataset.
N_SAMPLE_USERS = 150
MIN_EVENTS_PER_USER = 500  # drop very sparse users before sampling

SESSION_GAP_MINUTES = 30  # standard session-boundary threshold in listening-log literature

ROLLING_WEEKS = 4          # trailing window for rolling engagement features
CHURN_HORIZON_WEEKS = 4    # "went quiet for this many weeks" defines disengagement

RANDOM_SEED = 42
TEST_SIZE_USERS = 0.2

MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"
STREAM_STATE_DB = DATA_DIR / "stream_state.db"
