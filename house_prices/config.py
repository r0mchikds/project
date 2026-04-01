from pathlib import Path

# Основные настройки проекта
SEED = 42
TARGET_COL = "SalePrice"
ID_COL = "Id"

# Для финального пайплайна берём лучшую одиночную модель после feature engineering
USE_FEATURE_ENGINEERING = True

# Настройки кросс-валидации
N_SPLITS = 5

# Параметры лучшей модели
MODEL_PARAMS = {
    "depth": 6,
    "learning_rate": 0.03,
    "iterations": 1000,
    "l2_leaf_reg": 1,
    "random_state": SEED,
    "verbose": False,
    "allow_writing_files": False,
}

# Пути
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

SUBMISSION_DIR = PROJECT_DIR / "submissions"
SUBMISSION_PATH = SUBMISSION_DIR / "submission_catboost_post_fe.csv"
METRICS_PATH = SUBMISSION_DIR / "cv_metrics_catboost_post_fe.json"
