import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from typing import Dict, Optional, Tuple
from xgboost import XGBClassifier

RANDOM_STATE = 42
MIN_COMPLETE_SEASON_GAMES = 1000

PARAM_GRID = [
    {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
    },
    {
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.02,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 4,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
]


def season_game_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("SEASON").size().sort_index()


def choose_validation_and_holdout(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    counts = season_game_counts(df)
    seasons = counts.index.tolist()

    if len(seasons) < 3:
        raise ValueError("Se necesitan al menos 3 temporadas para entrenar, validar y probar.")

    last_season = seasons[-1]
    if counts[last_season] < MIN_COMPLETE_SEASON_GAMES:
        return seasons[-2], last_season

    return last_season, None


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_df = df.drop(columns=["DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM", "HOME_WIN"])
    target = df["HOME_WIN"]
    return feature_df, target


def fit_model(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series, params: dict) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    return model


def score_model(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba),
        "roc_auc": roc_auc_score(y, proba),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label} | "
        f"accuracy={metrics['accuracy']:.4f} | "
        f"log_loss={metrics['log_loss']:.4f} | "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )


def main() -> None:
    df = pd.read_csv("dataset_ml.csv")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(["DATE", "SEASON"]).reset_index(drop=True)

    validation_season, holdout_season = choose_validation_and_holdout(df)
    train_df = df[df["SEASON"] < validation_season].copy()
    valid_df = df[df["SEASON"] == validation_season].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("No hay suficientes datos para construir train y validacion temporal.")

    X_train, y_train = build_xy(train_df)
    X_valid, y_valid = build_xy(valid_df)

    best_params = None
    best_metrics = None
    best_score = float("inf")

    print(f"Temporada de validacion: {validation_season}")
    if holdout_season:
        print(f"Temporada holdout: {holdout_season}")

    for params in PARAM_GRID:
        model = fit_model(X_train, y_train, X_valid, y_valid, params)
        metrics = score_model(model, X_valid, y_valid)
        print_metrics(f"Validacion {params}", metrics)

        if metrics["log_loss"] < best_score:
            best_score = metrics["log_loss"]
            best_params = params
            best_metrics = metrics

    print("\nMejor configuracion:")
    print(best_params)
    print_metrics("Mejor validacion", best_metrics)

    final_train_df = df[df["SEASON"] <= validation_season].copy()
    X_final_train, y_final_train = build_xy(final_train_df)

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        **best_params,
    )
    final_model.fit(X_final_train, y_final_train, verbose=False)

    if holdout_season:
        holdout_df = df[df["SEASON"] == holdout_season].copy()
        X_holdout, y_holdout = build_xy(holdout_df)
        holdout_metrics = score_model(final_model, X_holdout, y_holdout)
        print_metrics("Holdout", holdout_metrics)

    importances = pd.Series(final_model.feature_importances_, index=X_final_train.columns)
    top_features = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 features:")
    for feature_name, importance in top_features.items():
        print(f"{feature_name}: {importance:.4f}")

    joblib.dump(
        {
            "model": final_model,
            "features": X_final_train.columns.tolist(),
            "best_params": best_params,
            "validation_season": validation_season,
            "holdout_season": holdout_season,
        },
        "nba_model.pkl",
    )

    print("\nModelo guardado como nba_model.pkl")


if __name__ == "__main__":
    main()
