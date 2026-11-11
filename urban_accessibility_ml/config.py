from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibility constants
SEED = 42
N_SPLITS = 5

# Dimensions and profiles
DIMENSIONS = [
    "pendiente",
    "anchura",
    "pavimento",
    "cruces",
    "senalizacion",
    "iluminacion",
    "mobiliario",
]
PROFILES = ["global", "funcional", "sensorial", "cognitiva"]

# Target columns
TARGET_COLUMNS = {
    "global": "clase_total",
    "funcional": "clase_funcional",
    "sensorial": "clase_sensorial",
    "cognitiva": "clase_cognitiva",
}

# Class order
CLASS_ORDER = ["C", "B", "A"]

# Viability (coming from the thesis)
VIABILITY = {
    "pendiente": {"cartografia": 25, "aerea": 0, "calle": 40},
    "anchura": {"cartografia": 15, "aerea": 45, "calle": 95},
    "pavimento": {"cartografia": 0, "aerea": 40, "calle": 30},
    "cruces": {"cartografia": 20, "aerea": 70, "calle": 50},
    "mobiliario": {"cartografia": 0, "aerea": 65, "calle": 80},
    "senalizacion": {"cartografia": 0, "aerea": 10, "calle": 50},
    "iluminacion": {"cartografia": 0, "aerea": 15, "calle": 30},
}

# Threshold for adding a dimension as viable
THRESHOLD = 50
# Thresholds of the sensitivity analysis (THRESHOLD is the main one)
THRESHOLDS = [40, THRESHOLD, 60, 75, 90]
# Refutation criterion of the thesis
CLEARLY_INFERIOR_RATIO = 0.8

# Feature sets
FEATURE_SETS = ["cartografico_aereo", "calle", "combinado", "completo"]

# Cities
CITIES = ["Albacete", "Algeciras"]

# Models of the experiment matrix
MATRIX_MODELS = ["dummy", "logistic_regression", "random_forest", "xgboost_balanced", "lightgbm"]

# Metrics
METRICS = ["f1_macro", "balanced_accuracy", "quadratic_weighted_kappa"]

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
