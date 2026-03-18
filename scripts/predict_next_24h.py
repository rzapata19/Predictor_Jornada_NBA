from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import requests

from build_dataset import (
    ROLLING_WINDOWS,
    TeamState,
    capped_rest,
    expected_result,
    safe_mean,
    season_regressed_elo,
    team_map,
    update_elo,
    window_mean,
)

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
MODEL_PATH = REPO_ROOT / "models" / "nba_model.pkl"
DATASET_PATH = REPO_ROOT / "data" / "dataset_ml.csv"
INPUT_SCHEDULE_PATH = REPO_ROOT / "data" / "remaining_schedule.csv"
OUTPUT_PATH = REPO_ROOT / "outputs" / "next_24h_predictions.csv"
TARGET_SEASON = "2025-26"
SCHEDULE_URLS = [
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_9.json",
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json",
]


@dataclass
class PredictionContext:
    states: Dict[str, TeamState]
    head_to_head: defaultdict[Tuple[str, str], Deque[Tuple[str, int]]]


def load_model_bundle() -> dict:
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("nba_model.pkl no tiene el formato esperado.")
    return bundle


def ensure_features(feature_row: dict, feature_names: List[str]) -> pd.DataFrame:
    for feature_name in feature_names:
        feature_row.setdefault(feature_name, 0.0)
    return pd.DataFrame([{feature_name: feature_row[feature_name] for feature_name in feature_names}])


def normalize_schedule_team_name(team_payload: dict) -> Optional[str]:
    city = str(team_payload.get("teamCity", "")).strip()
    name = str(team_payload.get("teamName", "")).strip()
    if not city and not name:
        return None

    full_name = " ".join(f"{city} {name}".split())
    return team_map.get(full_name, full_name)


def load_remaining_schedule(season: str, start_date: pd.Timestamp) -> pd.DataFrame:
    if INPUT_SCHEDULE_PATH.exists():
        schedule = pd.read_csv(INPUT_SCHEDULE_PATH)
        schedule["DATE"] = pd.to_datetime(schedule["DATE"])
        if "SEASON" not in schedule.columns:
            schedule["SEASON"] = season
        schedule = schedule[schedule["SEASON"] == season].copy()
        return schedule.sort_values(["DATE", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)

    payload = None
    last_error = None
    for url in SCHEDULE_URLS:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc

    if payload is None:
        raise FileNotFoundError(
            "No se pudo cargar el calendario restante. "
            "Usa remaining_schedule.csv con columnas DATE, HOME_TEAM, AWAY_TEAM, SEASON."
        ) from last_error

    rows: List[dict] = []
    seen = set()
    for date_block in payload.get("leagueSchedule", {}).get("gameDates", []):
        for game in date_block.get("games", []):
            season_year = str(game.get("seasonYear", "")).strip()
            if season_year and season_year != season:
                continue

            game_date_str = game.get("gameDate") or game.get("gameDateUTC") or date_block.get("gameDate")
            if not game_date_str:
                continue

            game_date = pd.to_datetime(game_date_str, errors="coerce")
            if pd.isna(game_date):
                continue
            game_date = pd.Timestamp(game_date)
            if game_date.tzinfo is not None:
                game_date = game_date.tz_convert(None)
            game_date = game_date.normalize()

            if game_date < start_date.normalize():
                continue

            status_text = str(game.get("gameStatusText", "")).lower()
            if "final" in status_text or "postponed" in status_text:
                continue

            home_team = normalize_schedule_team_name(game.get("homeTeam", {}))
            away_team = normalize_schedule_team_name(game.get("awayTeam", {}))
            if not home_team or not away_team:
                continue

            game_key = (game_date.date().isoformat(), home_team, away_team)
            if game_key in seen:
                continue

            seen.add(game_key)
            rows.append(
                {
                    "DATE": game_date,
                    "SEASON": season,
                    "HOME_TEAM": home_team,
                    "AWAY_TEAM": away_team,
                }
            )

    if not rows:
        raise FileNotFoundError("No se encontraron partidos futuros. Usa remaining_schedule.csv.")

    return pd.DataFrame(rows).sort_values(["DATE", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)


def add_window_features(row: dict, prefix: str, state: TeamState) -> None:
    for window in ROLLING_WINDOWS:
        row[f"{prefix}_WINRATE_LAST_{window}"] = window_mean(state.results, window, 0.5)
        row[f"{prefix}_MARGIN_LAST_{window}"] = window_mean(state.margin, window, 0.0)
        row[f"{prefix}_POINTS_FOR_LAST_{window}"] = window_mean(state.points_for, window, 110.0)
        row[f"{prefix}_POINTS_AGAINST_LAST_{window}"] = window_mean(state.points_against, window, 110.0)


def apply_historical_result(
    states: Dict[str, TeamState],
    head_to_head: defaultdict[Tuple[str, str], Deque[Tuple[str, int]]],
    game: pd.Series,
) -> None:
    home_team = game["HOME_TEAM"]
    away_team = game["AWAY_TEAM"]
    date = pd.Timestamp(game["DATE"])
    home_state = states[home_team]
    away_state = states[away_team]
    home_win = int(game["HOME_WIN"])

    probability_home = float(game["EXPECTED_HOME_WIN"]) if "EXPECTED_HOME_WIN" in game else 0.5
    margin = int(round((probability_home - 0.5) * 24))
    margin = max(1, abs(margin))
    home_points = 110 + margin if home_win else 110 - margin
    away_points = 110 - margin if home_win else 110 + margin

    new_home_elo, new_away_elo = update_elo(home_state.elo, away_state.elo, home_win, home_points, away_points)
    home_state.elo = new_home_elo
    away_state.elo = new_away_elo

    home_state.season_games += 1
    away_state.season_games += 1
    home_state.season_wins += home_win
    away_state.season_wins += 1 - home_win
    home_state.season_points_for += home_points
    home_state.season_points_against += away_points
    away_state.season_points_for += away_points
    away_state.season_points_against += home_points
    home_state.season_home_games += 1
    home_state.season_home_wins += home_win
    away_state.season_away_games += 1
    away_state.season_away_wins += 1 - home_win

    home_state.results.append(home_win)
    away_state.results.append(1 - home_win)
    home_state.margin.append(home_points - away_points)
    away_state.margin.append(away_points - home_points)
    home_state.points_for.append(home_points)
    home_state.points_against.append(away_points)
    away_state.points_for.append(away_points)
    away_state.points_against.append(home_points)
    home_state.home_results.append(home_win)
    away_state.away_results.append(1 - home_win)
    home_state.last_date = date
    away_state.last_date = date
    home_state.last_home_date = date
    away_state.last_away_date = date

    h2h_key = tuple(sorted((home_team, away_team)))
    head_to_head[h2h_key].append((home_team, home_win))


def build_current_context(target_season: str) -> Tuple[PredictionContext, pd.Timestamp]:
    games = pd.read_csv(DATASET_PATH)
    games["DATE"] = pd.to_datetime(games["DATE"])
    games = games.sort_values(["DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)

    season_team_sets = {
        season: set(
            pd.concat(
                [
                    games.loc[games["SEASON"] == season, "HOME_TEAM"],
                    games.loc[games["SEASON"] == season, "AWAY_TEAM"],
                ]
            ).unique()
        )
        for season in sorted(games["SEASON"].unique())
    }

    previous_states: Dict[str, TeamState] = {}
    states_by_season: Dict[str, Dict[str, TeamState]] = {}
    head_to_head: defaultdict[Tuple[str, str], Deque[Tuple[str, int]]] = defaultdict(lambda: deque(maxlen=5))

    for season in sorted(season_team_sets):
        season_states = season_regressed_elo(previous_states, season_team_sets[season])
        states_by_season[season] = season_states
        season_games = games[games["SEASON"] == season]

        for _, game in season_games.iterrows():
            apply_historical_result(season_states, head_to_head, game)

        previous_states = {team: TeamState(elo=state.elo) for team, state in season_states.items()}

    current_games = games[games["SEASON"] == target_season]
    if current_games.empty:
        raise ValueError(f"No se encontraron partidos para la temporada {target_season}.")

    latest_played_date = pd.Timestamp(current_games["DATE"].max())
    return PredictionContext(states_by_season[target_season], head_to_head), latest_played_date


def build_feature_row(context: PredictionContext, date: pd.Timestamp, home_team: str, away_team: str) -> dict:
    home_state = context.states[home_team]
    away_state = context.states[away_team]

    home_rest = capped_rest(date, home_state.last_date)
    away_rest = capped_rest(date, away_state.last_date)
    home_site_rest = capped_rest(date, home_state.last_home_date)
    away_site_rest = capped_rest(date, away_state.last_away_date)

    h2h_key = tuple(sorted((home_team, away_team)))
    h2h_history = context.head_to_head[h2h_key]
    home_h2h_winrate = safe_mean(
        deque([result if first_team == home_team else 1 - result for first_team, result in h2h_history], maxlen=5),
        0.5,
    )

    row = {
        "HOME_ELO": home_state.elo,
        "AWAY_ELO": away_state.elo,
        "ELO_DIFF": home_state.elo - away_state.elo,
        "EXPECTED_HOME_WIN": expected_result(home_state.elo, away_state.elo),
        "HOME_SEASON_WINRATE": (home_state.season_wins / home_state.season_games) if home_state.season_games else 0.5,
        "AWAY_SEASON_WINRATE": (away_state.season_wins / away_state.season_games) if away_state.season_games else 0.5,
        "SEASON_WINRATE_DIFF": (
            (home_state.season_wins / home_state.season_games) if home_state.season_games else 0.5
        ) - (
            (away_state.season_wins / away_state.season_games) if away_state.season_games else 0.5
        ),
        "SEASON_MARGIN_DIFF": (
            (home_state.season_points_for - home_state.season_points_against) / home_state.season_games
            if home_state.season_games
            else 0.0
        ) - (
            (away_state.season_points_for - away_state.season_points_against) / away_state.season_games
            if away_state.season_games
            else 0.0
        ),
        "SEASON_POINTS_FOR_DIFF": (
            home_state.season_points_for / home_state.season_games if home_state.season_games else 110.0
        ) - (
            away_state.season_points_for / away_state.season_games if away_state.season_games else 110.0
        ),
        "SEASON_POINTS_AGAINST_DIFF": (
            home_state.season_points_against / home_state.season_games if home_state.season_games else 110.0
        ) - (
            away_state.season_points_against / away_state.season_games if away_state.season_games else 110.0
        ),
        "HOME_HOME_WINRATE": (
            home_state.season_home_wins / home_state.season_home_games if home_state.season_home_games else 0.5
        ),
        "AWAY_AWAY_WINRATE": (
            away_state.season_away_wins / away_state.season_away_games if away_state.season_away_games else 0.5
        ),
        "SITE_WINRATE_DIFF": (
            (home_state.season_home_wins / home_state.season_home_games) if home_state.season_home_games else 0.5
        ) - (
            (away_state.season_away_wins / away_state.season_away_games) if away_state.season_away_games else 0.5
        ),
        "REST_DIFF": home_rest - away_rest,
        "HOME_REST_DAYS": home_rest,
        "AWAY_REST_DAYS": away_rest,
        "HOME_SITE_REST_DAYS": home_site_rest,
        "AWAY_SITE_REST_DAYS": away_site_rest,
        "HOME_B2B": int(home_rest <= 1),
        "AWAY_B2B": int(away_rest <= 1),
        "HOME_3IN4": int(home_rest <= 1 and home_site_rest <= 2),
        "AWAY_3IN4": int(away_rest <= 1 and away_site_rest <= 2),
        "H2H_HOME_WINRATE_LAST_5": home_h2h_winrate,
    }

    add_window_features(row, "HOME", home_state)
    add_window_features(row, "AWAY", away_state)

    for window in ROLLING_WINDOWS:
        row[f"WINRATE_DIFF_LAST_{window}"] = row[f"HOME_WINRATE_LAST_{window}"] - row[f"AWAY_WINRATE_LAST_{window}"]
        row[f"MARGIN_DIFF_LAST_{window}"] = row[f"HOME_MARGIN_LAST_{window}"] - row[f"AWAY_MARGIN_LAST_{window}"]
        row[f"POINTS_FOR_DIFF_LAST_{window}"] = row[f"HOME_POINTS_FOR_LAST_{window}"] - row[f"AWAY_POINTS_FOR_LAST_{window}"]
        row[f"POINTS_AGAINST_DIFF_LAST_{window}"] = (
            row[f"HOME_POINTS_AGAINST_LAST_{window}"] - row[f"AWAY_POINTS_AGAINST_LAST_{window}"]
        )

    return row


def format_prediction_row(game: pd.Series, probability_home: float) -> dict:
    home_team = game["HOME_TEAM"]
    away_team = game["AWAY_TEAM"]
    favorite = home_team if probability_home >= 0.5 else away_team
    favorite_prob = probability_home if probability_home >= 0.5 else 1 - probability_home

    return {
        "DATE": pd.Timestamp(game["DATE"]).strftime("%Y-%m-%d"),
        "HOME_TEAM": home_team,
        "AWAY_TEAM": away_team,
        "HOME_WIN_PROB": round(probability_home, 4),
        "AWAY_WIN_PROB": round(1 - probability_home, 4),
        "FAVORITE": favorite,
        "FAVORITE_WIN_PROB": round(favorite_prob, 4),
    }


def main() -> None:
    model_bundle = load_model_bundle()
    context, latest_played_date = build_current_context(TARGET_SEASON)
    schedule = load_remaining_schedule(TARGET_SEASON, latest_played_date)

    now = pd.Timestamp.now().tz_localize(None)
    window_end = now + pd.Timedelta(hours=24)
    next_games = schedule[(schedule["DATE"] >= now.normalize()) & (schedule["DATE"] <= window_end.normalize())].copy()
    next_games = next_games.sort_values(["DATE", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)

    if next_games.empty:
        print(f"No hay partidos programados entre {now:%Y-%m-%d %H:%M} y {window_end:%Y-%m-%d %H:%M}.")
        return

    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    prediction_rows = []

    print(f"Predicciones para las proximas 24 horas ({now:%Y-%m-%d %H:%M} -> {window_end:%Y-%m-%d %H:%M})")
    print("-" * 90)

    for _, game in next_games.iterrows():
        feature_row = build_feature_row(context, pd.Timestamp(game["DATE"]), game["HOME_TEAM"], game["AWAY_TEAM"])
        X = ensure_features(feature_row, feature_names)
        probability_home = float(model.predict_proba(X)[0, 1])
        prediction = format_prediction_row(game, probability_home)
        prediction_rows.append(prediction)

        print(
            f"{prediction['DATE']} | {prediction['AWAY_TEAM']} @ {prediction['HOME_TEAM']} | "
            f"P(local)={prediction['HOME_WIN_PROB']:.2%} | "
            f"Favorito: {prediction['FAVORITE']} ({prediction['FAVORITE_WIN_PROB']:.2%})"
        )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT_PATH, index=False)
    print("-" * 90)
    print(f"Predicciones guardadas en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
