from pathlib import Path

# Основные настройки проекта
SEED = 42
TARGET_COL = "Survived"
ID_COL = "PassengerId"

# Для финального пайплайна берём лучшую одиночную модель до feature engineering
USE_FEATURE_ENGINEERING = False

# Настройки кросс-валидации
N_SPLITS = 5
THRESHOLD = 0.5

# Базовые признаки
BASE_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "Pclass",
    "Sex",
    "Embarked",
]

BASE_NUMERICAL_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
]

BASE_CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
]

# Признаки после feature engineering
FE_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "CabinKnown",
    "FarePerPerson",
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
]

FE_NUMERICAL_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "CabinKnown",
    "FarePerPerson",
]

FE_CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
]

# Параметры лучшей модели
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "max_features": None,
    "class_weight": None,
    "random_state": SEED,
}

# Пути
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

SUBMISSION_DIR = PROJECT_DIR / "submissions"
SUBMISSION_PATH = SUBMISSION_DIR / "submission_random_forest_pre_fe.csv"
METRICS_PATH = SUBMISSION_DIR / "cv_metrics_random_forest_pre_fe.json"
