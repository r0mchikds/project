from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import (
        BASE_CATEGORICAL_FEATURES,
        BASE_FEATURES,
        BASE_NUMERICAL_FEATURES,
        FE_CATEGORICAL_FEATURES,
        FE_FEATURES,
        FE_NUMERICAL_FEATURES,
        ID_COL,
        METRICS_PATH,
        MODEL_PARAMS,
        N_SPLITS,
        SEED,
        SUBMISSION_DIR,
        SUBMISSION_PATH,
        TARGET_COL,
        TEST_PATH,
        THRESHOLD,
        TRAIN_PATH,
        USE_FEATURE_ENGINEERING,
    )
except ImportError:
    from config import (
        BASE_CATEGORICAL_FEATURES,
        BASE_FEATURES,
        BASE_NUMERICAL_FEATURES,
        FE_CATEGORICAL_FEATURES,
        FE_FEATURES,
        FE_NUMERICAL_FEATURES,
        ID_COL,
        METRICS_PATH,
        MODEL_PARAMS,
        N_SPLITS,
        SEED,
        SUBMISSION_DIR,
        SUBMISSION_PATH,
        TARGET_COL,
        TEST_PATH,
        THRESHOLD,
        TRAIN_PATH,
        USE_FEATURE_ENGINEERING,
    )


def seed_everything(seed: int = 42) -> None:
    """Фиксирует сиды для воспроизводимости."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except ImportError:
        pass


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет простые признаки, которые использовались в ноутбуке."""
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Miss",
        }
    )

    rare_titles = [
        "Lady",
        "Countess",
        "the Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает train и test."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def get_feature_lists() -> tuple[list[str], list[str], list[str]]:
    """Возвращает списки признаков для выбранного режима."""
    if USE_FEATURE_ENGINEERING:
        return FE_FEATURES, FE_NUMERICAL_FEATURES, FE_CATEGORICAL_FEATURES

    return BASE_FEATURES, BASE_NUMERICAL_FEATURES, BASE_CATEGORICAL_FEATURES


def build_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Собирает пайплайн препроцессинга и модели."""
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(**MODEL_PARAMS)),
        ]
    )

    return pipeline


def run_cv_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> dict:
    """Считает метрики на Stratified 5-Fold CV и усредняет предсказания на test."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_proba = np.zeros(X.shape[0], dtype=np.float32)
    test_fold_probs = []

    fold_accuracy = []
    fold_f1 = []
    fold_auc = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model = clone(pipeline)
        model.fit(X_train_fold, y_train_fold)

        val_proba = model.predict_proba(X_val_fold)[:, 1]
        val_pred = (val_proba >= THRESHOLD).astype(int)

        test_proba = model.predict_proba(X_test)[:, 1]
        test_fold_probs.append(test_proba)

        oof_proba[val_idx] = val_proba

        fold_accuracy.append(accuracy_score(y_val_fold, val_pred))
        fold_f1.append(f1_score(y_val_fold, val_pred))
        fold_auc.append(roc_auc_score(y_val_fold, val_proba))

    oof_pred = (oof_proba >= THRESHOLD).astype(int)
    mean_test_proba = np.mean(test_fold_probs, axis=0)
    test_pred = (mean_test_proba >= THRESHOLD).astype(int)

    results = {
        "fold_accuracy": fold_accuracy,
        "fold_f1": fold_f1,
        "fold_auc": fold_auc,
        "oof_accuracy": float(accuracy_score(y, oof_pred)),
        "oof_f1": float(f1_score(y, oof_pred)),
        "oof_auc": float(roc_auc_score(y, oof_proba)),
        "mean_fold_accuracy": float(np.mean(fold_accuracy)),
        "std_fold_accuracy": float(np.std(fold_accuracy)),
        "mean_fold_f1": float(np.mean(fold_f1)),
        "std_fold_f1": float(np.std(fold_f1)),
        "mean_fold_auc": float(np.mean(fold_auc)),
        "std_fold_auc": float(np.std(fold_auc)),
        "test_pred": test_pred,
    }

    return results


def save_submission(test_ids: pd.Series, preds: np.ndarray) -> None:
    """Сохраняет submission-файл."""
    submission = pd.DataFrame(
        {
            ID_COL: test_ids,
            TARGET_COL: preds.astype(int),
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)


def save_metrics(metrics: dict) -> None:
    """Сохраняет метрики в json."""
    metrics_to_save = {
        "model_name": "RandomForestClassifier",
        "use_feature_engineering": USE_FEATURE_ENGINEERING,
        "model_params": MODEL_PARAMS,
        "cv_metrics": {
            "mean_fold_accuracy": metrics["mean_fold_accuracy"],
            "std_fold_accuracy": metrics["std_fold_accuracy"],
            "mean_fold_f1": metrics["mean_fold_f1"],
            "std_fold_f1": metrics["std_fold_f1"],
            "mean_fold_auc": metrics["mean_fold_auc"],
            "std_fold_auc": metrics["std_fold_auc"],
            "oof_accuracy": metrics["oof_accuracy"],
            "oof_f1": metrics["oof_f1"],
            "oof_auc": metrics["oof_auc"],
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)


def main() -> dict:
    """Основной запуск проекта."""
    seed_everything(SEED)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_data()

    if USE_FEATURE_ENGINEERING:
        train = add_basic_features(train)
        test = add_basic_features(test)

    features, numerical_features, categorical_features = get_feature_lists()

    X = train[features].copy()
    y = train[TARGET_COL].copy()
    X_test = test[features].copy()

    pipeline = build_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    cv_results = run_cv_pipeline(
        pipeline=pipeline,
        X=X,
        y=y,
        X_test=X_test,
    )

    # Итоговый submission формируем по усреднённым предсказаниям моделей с CV
    final_test_pred = cv_results["test_pred"]

    save_submission(test_ids=test[ID_COL], preds=final_test_pred)
    save_metrics(cv_results)

    print("Titanic pipeline finished")
    print(f"Mean CV Accuracy: {cv_results['mean_fold_accuracy']:.4f}")
    print(f"Mean CV F1:       {cv_results['mean_fold_f1']:.4f}")
    print(f"Mean CV ROC AUC:  {cv_results['mean_fold_auc']:.4f}")
    print(f"Submission saved to: {SUBMISSION_PATH}")
    print(f"Metrics saved to:    {METRICS_PATH}")

    return {
        "project": "Titanic",
        "model": "RandomForestClassifier",
        "mean_cv_accuracy": cv_results["mean_fold_accuracy"],
        "mean_cv_f1": cv_results["mean_fold_f1"],
        "mean_cv_auc": cv_results["mean_fold_auc"],
        "submission_path": str(SUBMISSION_PATH),
        "metrics_path": str(METRICS_PATH),
    }


if __name__ == "__main__":
    main()
