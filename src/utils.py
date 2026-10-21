import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
NEWS_CORPUS_DIR = RAW_DATA_DIR / "news_corpus"
SATELLITE_DIR = RAW_DATA_DIR / "satellite"
SENTINEL1_DIR = SATELLITE_DIR / "sentinel1"
SENTINEL2_DIR = SATELLITE_DIR / "sentinel2"
DEM_DIR = SATELLITE_DIR / "dem"
RAINFALL_DIR = RAW_DATA_DIR / "rainfall"

INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_STACK_DIR = PROCESSED_DATA_DIR / "feature_stack"

GAZETTEER_DIR = BASE_DIR / "gazetteer"
MODELS_DIR = BASE_DIR / "models"
CLASSIFIER_MODEL_DIR = MODELS_DIR / "classifier"
LORA_TABULAR_MODEL_DIR = MODELS_DIR / "lora_tabular"
GEOFM_LORA_MODEL_DIR = LORA_TABULAR_MODEL_DIR  # Backward-compatibility alias

SRC_DIR = BASE_DIR / "src"
OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

ALL_DIRS = [
    DATA_DIR, RAW_DATA_DIR, NEWS_CORPUS_DIR, SATELLITE_DIR,
    SENTINEL1_DIR, SENTINEL2_DIR, DEM_DIR, RAINFALL_DIR,
    INTERIM_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_STACK_DIR,
    GAZETTEER_DIR, MODELS_DIR, CLASSIFIER_MODEL_DIR, GEOFM_LORA_MODEL_DIR,
    SRC_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR
]


def ensure_directories():
    """Ensure all required directories exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name="araku_landslide"):
    """Get a standardized logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def is_offline_mode():
    """Check if offline fallback mode is enabled.
    Defaults to False (live mode). Set OFFLINE_MODE=true in .env to force offline.
    """
    val = os.getenv("OFFLINE_MODE", "false").lower()
    return val in ("true", "1", "yes")


if __name__ == "__main__":
    ensure_directories()
    print("Project directory structure verified successfully.")
