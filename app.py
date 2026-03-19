from pathlib import Path
import os
import subprocess
import sys

from flask import Flask, render_template, request
import pandas as pd

from scripts.predict_next_24h import generate_predictions_for_date, get_available_prediction_dates


PROJECT_ROOT = Path(__file__).resolve().parent
INJURIES_PATH = PROJECT_ROOT / "data" / "injuries_current.csv"
DAILY_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "daily_predictions"
DAILY_UPDATE_PATH = PROJECT_ROOT / "scripts" / "daily_update.py"

app = Flask(__name__)


def load_injuries_table() -> pd.DataFrame:
    if not INJURIES_PATH.exists():
        return pd.DataFrame()
    injuries_df = pd.read_csv(INJURIES_PATH)
    return injuries_df.sort_values(["TEAM", "STATUS", "PLAYER"]).reset_index(drop=True)


def load_daily_history() -> list[str]:
    if not DAILY_OUTPUT_DIR.exists():
        return []
    return [file.name for file in sorted(DAILY_OUTPUT_DIR.glob("predictions_*.csv"), reverse=True)]


def build_recent_accuracy_summary(limit: int = 7) -> list[dict]:
    rows = []
    historical_dates = sorted(get_available_prediction_dates())
    today = pd.Timestamp.now().normalize()

    for day in reversed(historical_dates):
        timestamp = pd.Timestamp(day).normalize()
        if timestamp >= today:
            continue

        predictions_df = generate_predictions_for_date(timestamp)
        if predictions_df.empty or "ACTUAL_WINNER" not in predictions_df.columns:
            continue

        resolved = predictions_df["ACTUAL_WINNER"].astype(str).str.len().gt(0)
        if not resolved.any():
            continue

        evaluated = predictions_df[resolved].copy()
        total_games = len(evaluated)
        hits = int(evaluated["PREDICTION_HIT"].fillna(0).astype(int).sum())
        rows.append(
            {
                "date_label": timestamp.strftime("%d/%m"),
                "date_value": timestamp.strftime("%Y-%m-%d"),
                "hits": hits,
                "total_games": total_games,
                "accuracy": round(hits / total_games, 4) if total_games else 0.0,
            }
        )

        if len(rows) >= limit:
            break

    return rows


def build_day_options() -> list[dict]:
    options = []
    for value in get_available_prediction_dates():
        timestamp = pd.Timestamp(value).normalize()
        options.append(
            {
                "value": timestamp.strftime("%Y-%m-%d"),
                "label": timestamp.strftime("%d/%m"),
            }
        )
    return options


def resolve_selected_day(day_options: list[dict]) -> pd.Timestamp:
    raw_day = request.args.get("day", "").strip()
    available_values = {option["value"] for option in day_options}

    if raw_day:
        try:
            parsed_day = pd.Timestamp(raw_day).normalize()
            if parsed_day.strftime("%Y-%m-%d") in available_values:
                return parsed_day
        except ValueError:
            pass

    today = pd.Timestamp.now().normalize()
    if today.strftime("%Y-%m-%d") in available_values:
        return today

    if not day_options:
        return today
    return pd.Timestamp(day_options[-1]["value"]).normalize()


def run_daily_update() -> None:
    if not DAILY_UPDATE_PATH.exists():
        raise FileNotFoundError(f"No se encontro el script de actualizacion: {DAILY_UPDATE_PATH}")

    subprocess.run([sys.executable, str(DAILY_UPDATE_PATH)], cwd=PROJECT_ROOT, check=True)


@app.route("/")
def index():
    day_options = build_day_options()
    selected_day = resolve_selected_day(day_options)
    predictions_df = generate_predictions_for_date(selected_day)
    injuries_df = load_injuries_table()
    recent_accuracy = build_recent_accuracy_summary()
    selected_day_label = selected_day.strftime("%d/%m/%Y")
    resolved_mask = (
        predictions_df["ACTUAL_WINNER"].astype(str).str.len().gt(0)
        if "ACTUAL_WINNER" in predictions_df.columns
        else pd.Series(dtype=bool)
    )
    resolved_games = int(resolved_mask.sum()) if not resolved_mask.empty else 0
    is_historical = resolved_games > 0

    if is_historical:
        evaluated_df = predictions_df[resolved_mask].copy()
        hits = int(evaluated_df["PREDICTION_HIT"].fillna(0).astype(int).sum())
        total_games = len(predictions_df)
        hit_rate = round(hits / resolved_games, 4) if resolved_games else None
    else:
        hits = None
        total_games = len(predictions_df)
        hit_rate = None

    return render_template(
        "index.html",
        predictions=predictions_df.to_dict(orient="records"),
        injuries=injuries_df.to_dict(orient="records"),
        recent_accuracy=recent_accuracy,
        day_options=day_options,
        selected_day=selected_day.strftime("%Y-%m-%d"),
        selected_day_label=selected_day_label,
        is_historical=is_historical,
        resolved_games=resolved_games,
        hits=hits,
        total_games=total_games,
        hit_rate=hit_rate,
    )


if __name__ == "__main__":
    debug_mode = True
    should_run_update = not debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if should_run_update:
        try:
            run_daily_update()
        except Exception as exc:
            print(f"[WARN] daily_update fallo al iniciar la app: {exc}")
    app.run(debug=debug_mode)
