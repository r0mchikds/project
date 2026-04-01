from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

try:
    from .config import (
        ID_COL,
        METRICS_PATH,
        MODEL_PARAMS,
        N_SPLITS,
        SEED,
        SUBMISSION_DIR,
        SUBMISSION_PATH,
        TARGET_COL,
        TEST_PATH,
        TRAIN_PATH,
        USE_FEATURE_ENGINEERING,
    )
except ImportError:
    from config import (
        ID_COL,
        METRICS_PATH,
        MODEL_PARAMS,
        N_SPLITS,
        SEED,
        SUBMISSION_DIR,
        SUBMISSION_PATH,
        TARGET_COL,
        TEST_PATH,
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
    """Добавляет признаки, которые использовались в post-FE версии ноутбука."""
    df = df.copy()

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)

    df["HasGarage"] = df["GarageArea"].fillna(0).gt(0).astype(int)
    df["GarageAge"] = np.where(
        df["GarageYrBlt"].notna(),
        df["YrSold"] - df["GarageYrBlt"],
        0,
    )

    df["HasBsmt"] = df["TotalBsmtSF"].fillna(0).gt(0).astype(int)
    df["HasFireplace"] = df["Fireplaces"].fillna(0).gt(0).astype(int)
    df["Has2ndFloor"] = df["2ndFlrSF"].fillna(0).gt(0).astype(int)
    df["HasPool"] = df["PoolArea"].fillna(0).gt(0).astype(int)

    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    df["TotalOutdoorSF"] = (
        df["WoodDeckSF"].fillna(0)
        + df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )

    df["LivLotRatio"] = np.where(
        df["LotArea"].fillna(0) > 0,
        df["GrLivArea"].fillna(0) / df["LotArea"].fillna(1),
        0,
    )

    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]

    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает train и test."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Делит признаки на числовые и категориальные."""
    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

    if "MSSubClass" in numerical_features:
        numerical_features.remove("MSSubClass")
    if "MSSubClass" not in categorical_features and "MSSubClass" in X.columns:
        categorical_features.append("MSSubClass")

    categorical_features = sorted(categorical_features)
    numerical_features = sorted(numerical_features)

    return numerical_features, categorical_features


def preprocess_for_catboost(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Заполняет пропуски статистиками по train и приводит категории к строкам."""
    X_train = X_train.copy()
    X_other = X_other.copy()

    train_median = X_train[numerical_features].median()
    train_mode = X_train[categorical_features].mode().iloc[0]

    X_train[numerical_features] = X_train[numerical_features].fillna(train_median)
    X_other[numerical_features] = X_other[numerical_features].fillna(train_median)

    X_train[categorical_features] = X_train[categorical_features].fillna(train_mode)
    X_other[categorical_features] = X_other[categorical_features].fillna(train_mode)

    for col in categorical_features:
        X_train[col] = X_train[col].astype(str)
        X_other[col] = X_other[col].astype(str)

    return X_train, X_other


def run_cv_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
) -> dict:
    """Считает метрики на обычной 5-Fold CV и усредняет предсказания на test."""
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_pred = np.zeros(X.shape[0], dtype=np.float32)
    test_fold_pred = []
    fold_rmse = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        X_train_fold, X_val_fold = preprocess_for_catboost(
            X_train=X_train_fold,
            X_other=X_val_fold,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        _, X_test_fold = preprocess_for_catboost(
            X_train=X.iloc[train_idx].copy(),
            X_other=X_test.copy(),
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        model = CatBoostRegressor(
            **MODEL_PARAMS,
            cat_features=categorical_features,
        )
        model.fit(X_train_fold, y_train_fold)

        val_pred = model.predict(X_val_fold)
        test_pred = model.predict(X_test_fold)

        oof_pred[val_idx] = val_pred
        test_fold_pred.append(test_pred)

        fold_rmse.append(root_mean_squared_error(y_val_fold, val_pred))

    mean_test_pred = np.mean(test_fold_pred, axis=0)

    results = {
        "fold_rmse": fold_rmse,
        "oof_rmse": float(root_mean_squared_error(y, oof_pred)),
        "mean_fold_rmse": float(np.mean(fold_rmse)),
        "std_fold_rmse": float(np.std(fold_rmse)),
        "mean_test_pred": mean_test_pred,
    }

    return results


def save_submission(test_ids: pd.Series, preds_log: np.ndarray) -> None:
    """Сохраняет submission-файл."""
    sale_price_pred = np.exp(preds_log)

    submission = pd.DataFrame(
        {
            ID_COL: test_ids,
            TARGET_COL: sale_price_pred,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)


def save_metrics(metrics: dict) -> None:
    """Сохраняет метрики в json."""
    metrics_to_save = {
        "model_name": "CatBoostRegressor",
        "use_feature_engineering": USE_FEATURE_ENGINEERING,
        "model_params": MODEL_PARAMS,
        "cv_metrics": {
            "mean_fold_rmse": metrics["mean_fold_rmse"],
            "std_fold_rmse": metrics["std_fold_rmse"],
            "oof_rmse": metrics["oof_rmse"],
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

    feature_cols = [col for col in train.columns if col not in [ID_COL, TARGET_COL]]

    X = train[feature_cols].copy()
    y = np.log(train[TARGET_COL].copy())
    X_test = test[feature_cols].copy()

    numerical_features, categorical_features = get_feature_types(X)

    cv_results = run_cv_catboost(
        X=X,
        y=y,
        X_test=X_test,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # Итоговый submission формируем по усреднённым предсказаниям моделей с CV
    final_test_pred_log = cv_results["mean_test_pred"]

    save_submission(test_ids=test[ID_COL], preds_log=final_test_pred_log)
    save_metrics(cv_results)

    print("House Prices pipeline finished")
    print(f"Mean CV RMSE: {cv_results['mean_fold_rmse']:.4f}")
    print(f"OOF RMSE:     {cv_results['oof_rmse']:.4f}")
    print(f"Submission saved to: {SUBMISSION_PATH}")
    print(f"Metrics saved to:    {METRICS_PATH}")

    return {
        "project": "House Prices",
        "model": "CatBoostRegressor",
        "mean_cv_rmse": cv_results["mean_fold_rmse"],
        "oof_rmse": cv_results["oof_rmse"],
        "submission_path": str(SUBMISSION_PATH),
        "metrics_path": str(METRICS_PATH),
    }


if __name__ == "__main__":
    main()
