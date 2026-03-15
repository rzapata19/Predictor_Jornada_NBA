from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import random
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
MODEL_PATH = PROJECT_ROOT / "nba_model.pkl"
DATASET_PATH = PROJECT_ROOT / "dataset_ml.csv"
OUTPUT_DIR = PROJECT_ROOT / "simulation_outputs"
INPUT_SCHEDULE_PATH = PROJECT_ROOT / "remaining_schedule.csv"
SCHEDULE_TEMPLATE_PATH = PROJECT_ROOT / "remaining_schedule_template.csv"
TARGET_SEASON = "2025-26"
REGULAR_SEASON_END_DATE = "2026-04-20"
DEFAULT_SIMULATION_RUNS = 10
RANDOM_SEED = 42
DEFAULT_PLAYOFF_REST_DAYS = 2
SERIES_HOME_PATTERN = ("HIGH", "HIGH", "LOW", "LOW", "HIGH", "LOW", "HIGH")
SCHEDULE_URLS = [
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_9.json",
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json",
]

TEAM_CONFERENCES = {
    "Atlanta Hawks": "East",
    "Boston Celtics": "East",
    "Brooklyn Nets": "East",
    "Charlotte Hornets": "East",
    "Chicago Bulls": "East",
    "Cleveland Cavaliers": "East",
    "Detroit Pistons": "East",
    "Indiana Pacers": "East",
    "Miami Heat": "East",
    "Milwaukee Bucks": "East",
    "New York Knicks": "East",
    "Orlando Magic": "East",
    "Philadelphia 76ers": "East",
    "Toronto Raptors": "East",
    "Washington Wizards": "East",
    "Dallas Mavericks": "West",
    "Denver Nuggets": "West",
    "Golden State Warriors": "West",
    "Houston Rockets": "West",
    "Los Angeles Clippers": "West",
    "Los Angeles Lakers": "West",
    "Memphis Grizzlies": "West",
    "Minnesota Timberwolves": "West",
    "New Orleans Pelicans": "West",
    "Oklahoma City Thunder": "West",
    "Phoenix Suns": "West",
    "Portland Trail Blazers": "West",
    "Sacramento Kings": "West",
    "San Antonio Spurs": "West",
    "Utah Jazz": "West",
}

ROUND_NAME_BY_WINS = {
    4: "Champion",
    3: "Finals",
    2: "Conference Finals",
    1: "Conference Semifinals",
    0: "Conference Quarterfinals",
}


@dataclass
class SimulationContext:
    states: Dict[str, TeamState]
    head_to_head: defaultdict[Tuple[str, str], Deque[Tuple[str, int]]]
    current_date: pd.Timestamp
    current_season: str


def load_model_bundle() -> dict:
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("nba_model.pkl no tiene el formato esperado.")
    return bundle


def load_remaining_schedule(season: str, start_date: pd.Timestamp) -> pd.DataFrame:
    if INPUT_SCHEDULE_PATH.exists():
        schedule = pd.read_csv(INPUT_SCHEDULE_PATH)
        schedule["DATE"] = pd.to_datetime(schedule["DATE"])
        if "SEASON" not in schedule.columns:
            schedule["SEASON"] = season
        schedule = schedule[schedule["SEASON"] == season].copy()
        schedule = schedule[schedule["DATE"] > start_date].copy()
        return schedule.sort_values(["DATE", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)

    try:
        return fetch_schedule_from_api(season, start_date)
    except requests.exceptions.RequestException as exc:
        write_schedule_template(start_date + pd.Timedelta(days=1), season)
        raise FileNotFoundError(
            "No se pudo descargar el calendario restante desde stats.nba.com. "
            f"Se ha creado la plantilla {SCHEDULE_TEMPLATE_PATH.name}. "
            "Rellenala con columnas DATE, HOME_TEAM, AWAY_TEAM, SEASON y guardala como remaining_schedule.csv."
        ) from exc


def write_schedule_template(start_date: pd.Timestamp, season: str) -> None:
    template = pd.DataFrame(
        [
            {
                "DATE": start_date.strftime("%Y-%m-%d"),
                "HOME_TEAM": "Boston Celtics",
                "AWAY_TEAM": "Miami Heat",
                "SEASON": season,
            },
            {
                "DATE": (start_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "HOME_TEAM": "Los Angeles Lakers",
                "AWAY_TEAM": "Denver Nuggets",
                "SEASON": season,
            },
        ]
    )
    template.to_csv(SCHEDULE_TEMPLATE_PATH, index=False)


def fetch_schedule_from_api(season: str, start_date: pd.Timestamp) -> pd.DataFrame:
    end_date = pd.Timestamp(REGULAR_SEASON_END_DATE)
    if end_date <= start_date:
        return pd.DataFrame(columns=["DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM"])

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
        raise requests.exceptions.RequestException("No se pudo descargar el schedule oficial.") from last_error

    league_schedule = payload.get("leagueSchedule", {})
    game_dates = league_schedule.get("gameDates", [])
    rows: List[dict] = []
    seen = set()

    for date_block in game_dates:
        games = date_block.get("games", [])
        for game in games:
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

            if game_date <= start_date or game_date > end_date:
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
        raise FileNotFoundError(
            "No se encontraron partidos futuros en el schedule oficial. "
            "Crea remaining_schedule.csv con columnas DATE, HOME_TEAM, AWAY_TEAM, SEASON."
        )

    return pd.DataFrame(rows).sort_values(["DATE", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)


def normalize_schedule_team_name(team_payload: dict) -> Optional[str]:
    city = str(team_payload.get("teamCity", "")).strip()
    name = str(team_payload.get("teamName", "")).strip()
    if not city and not name:
        return None

    full_name = f"{city} {name}".strip()
    full_name = " ".join(full_name.split())
    return team_map.get(full_name, full_name)


def team_state_to_dict(team: str, state: TeamState) -> dict:
    return {
        "TEAM": team,
        "WINS": state.season_wins,
        "LOSSES": max(state.season_games - state.season_wins, 0),
        "WIN_PCT": (state.season_wins / state.season_games) if state.season_games else 0.0,
        "POINT_DIFF": (
            (state.season_points_for - state.season_points_against) / state.season_games if state.season_games else 0.0
        ),
        "ELO": state.elo,
        "HOME_WIN_PCT": (state.season_home_wins / state.season_home_games) if state.season_home_games else 0.0,
        "AWAY_WIN_PCT": (state.season_away_wins / state.season_away_games) if state.season_away_games else 0.0,
        "CONFERENCE": TEAM_CONFERENCES.get(team, "Unknown"),
    }


def add_window_features(row: dict, prefix: str, state: TeamState) -> None:
    for window in ROLLING_WINDOWS:
        row[f"{prefix}_WINRATE_LAST_{window}"] = window_mean(state.results, window, 0.5)
        row[f"{prefix}_MARGIN_LAST_{window}"] = window_mean(state.margin, window, 0.0)
        row[f"{prefix}_POINTS_FOR_LAST_{window}"] = window_mean(state.points_for, window, 110.0)
        row[f"{prefix}_POINTS_AGAINST_LAST_{window}"] = window_mean(state.points_against, window, 110.0)


def build_feature_row(
    context: SimulationContext,
    date: pd.Timestamp,
    home_team: str,
    away_team: str,
) -> dict:
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
        row[f"POINTS_FOR_DIFF_LAST_{window}"] = (
            row[f"HOME_POINTS_FOR_LAST_{window}"] - row[f"AWAY_POINTS_FOR_LAST_{window}"]
        )
        row[f"POINTS_AGAINST_DIFF_LAST_{window}"] = (
            row[f"HOME_POINTS_AGAINST_LAST_{window}"] - row[f"AWAY_POINTS_AGAINST_LAST_{window}"]
        )

    return row


def ensure_features(feature_row: dict, feature_names: List[str]) -> pd.DataFrame:
    for feature_name in feature_names:
        feature_row.setdefault(feature_name, 0.0)
    return pd.DataFrame([{feature_name: feature_row[feature_name] for feature_name in feature_names}])


def apply_game_result(
    context: SimulationContext,
    date: pd.Timestamp,
    home_team: str,
    away_team: str,
    home_win: int,
    probability_home: float,
    home_points: Optional[int] = None,
    away_points: Optional[int] = None,
) -> None:
    home_state = context.states[home_team]
    away_state = context.states[away_team]

    if home_points is None or away_points is None:
        # Convert the model probability into a synthetic point margin so rolling stats still move.
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
    context.head_to_head[h2h_key].append((home_team, home_win))
    context.current_date = max(context.current_date, date)


def build_current_context(target_season: str) -> Tuple[SimulationContext, pd.Timestamp]:
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
            context = SimulationContext(season_states, head_to_head, game["DATE"], season)
            expected_home = float(game["EXPECTED_HOME_WIN"]) if "EXPECTED_HOME_WIN" in game else 0.5
            apply_game_result(
                context=context,
                date=game["DATE"],
                home_team=game["HOME_TEAM"],
                away_team=game["AWAY_TEAM"],
                home_win=int(game["HOME_WIN"]),
                probability_home=expected_home,
            )

        previous_states = {team: TeamState(elo=state.elo) for team, state in season_states.items()}

    current_games = games[games["SEASON"] == target_season]
    if current_games.empty:
        raise ValueError(f"No se encontraron partidos jugados para la temporada {target_season}.")

    current_date = current_games["DATE"].max()
    return SimulationContext(states_by_season[target_season], head_to_head, current_date, target_season), current_date


def simulate_remaining_regular_season(
    context: SimulationContext,
    remaining_schedule: pd.DataFrame,
    model_bundle: dict,
    rng: random.Random,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    simulated_rows = []

    for _, game in remaining_schedule.sort_values(["DATE", "HOME_TEAM", "AWAY_TEAM"]).iterrows():
        feature_row = build_feature_row(context, game["DATE"], game["HOME_TEAM"], game["AWAY_TEAM"])
        X = ensure_features(feature_row, feature_names)
        probability_home = float(model.predict_proba(X)[0, 1])
        home_win = int(rng.random() < probability_home)
        apply_game_result(context, game["DATE"], game["HOME_TEAM"], game["AWAY_TEAM"], home_win, probability_home)

        simulated_rows.append(
            {
                "DATE": game["DATE"],
                "SEASON": game["SEASON"],
                "HOME_TEAM": game["HOME_TEAM"],
                "AWAY_TEAM": game["AWAY_TEAM"],
                "HOME_WIN_PROB": probability_home,
                "WINNER": game["HOME_TEAM"] if home_win else game["AWAY_TEAM"],
                "LOSER": game["AWAY_TEAM"] if home_win else game["HOME_TEAM"],
            }
        )

    standings = pd.DataFrame([team_state_to_dict(team, state) for team, state in context.states.items()])
    standings = standings.sort_values(["CONFERENCE", "WINS", "POINT_DIFF", "ELO", "TEAM"], ascending=[True, False, False, False, True])
    standings["SEED"] = standings.groupby("CONFERENCE").cumcount() + 1

    return standings.reset_index(drop=True), pd.DataFrame(simulated_rows)


def get_conference_standings(standings: pd.DataFrame, conference: str) -> pd.DataFrame:
    conference_table = standings[standings["CONFERENCE"] == conference].copy()
    return conference_table.sort_values(["WINS", "POINT_DIFF", "ELO", "TEAM"], ascending=[False, False, False, True]).reset_index(drop=True)


def simulate_single_game(
    context: SimulationContext,
    model_bundle: dict,
    date: pd.Timestamp,
    home_team: str,
    away_team: str,
    rng: random.Random,
) -> Tuple[str, float]:
    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    feature_row = build_feature_row(context, date, home_team, away_team)
    X = ensure_features(feature_row, feature_names)
    probability_home = float(model.predict_proba(X)[0, 1])
    home_win = int(rng.random() < probability_home)
    apply_game_result(context, date, home_team, away_team, home_win, probability_home)
    return (home_team if home_win else away_team), probability_home


def simulate_playin_for_conference(
    context: SimulationContext,
    model_bundle: dict,
    standings: pd.DataFrame,
    conference: str,
    rng: random.Random,
) -> Tuple[List[str], List[dict]]:
    table = get_conference_standings(standings, conference)
    top_six = table.head(6)["TEAM"].tolist()
    seeds_7_10 = table.iloc[6:10]["TEAM"].tolist()
    if len(seeds_7_10) < 4:
        raise ValueError(f"No hay suficientes equipos para play-in en {conference}.")

    day_1 = context.current_date + pd.Timedelta(days=1)
    winner_7_8, _ = simulate_single_game(context, model_bundle, day_1, seeds_7_10[0], seeds_7_10[1], rng)
    loser_7_8 = seeds_7_10[1] if winner_7_8 == seeds_7_10[0] else seeds_7_10[0]

    winner_9_10, _ = simulate_single_game(context, model_bundle, day_1, seeds_7_10[2], seeds_7_10[3], rng)

    day_2 = day_1 + pd.Timedelta(days=1)
    eighth_seed, _ = simulate_single_game(context, model_bundle, day_2, loser_7_8, winner_9_10, rng)

    final_seeds = top_six + [winner_7_8, eighth_seed]
    playin_rows = [
        {"CONFERENCE": conference, "GAME": "7v8", "HOME_TEAM": seeds_7_10[0], "AWAY_TEAM": seeds_7_10[1], "WINNER": winner_7_8},
        {"CONFERENCE": conference, "GAME": "9v10", "HOME_TEAM": seeds_7_10[2], "AWAY_TEAM": seeds_7_10[3], "WINNER": winner_9_10},
        {"CONFERENCE": conference, "GAME": "8-seed", "HOME_TEAM": loser_7_8, "AWAY_TEAM": winner_9_10, "WINNER": eighth_seed},
    ]
    context.current_date = day_2
    return final_seeds, playin_rows


def simulate_series(
    context: SimulationContext,
    model_bundle: dict,
    high_seed_team: str,
    low_seed_team: str,
    round_name: str,
    conference: str,
    rng: random.Random,
) -> Tuple[str, List[dict]]:
    wins = {high_seed_team: 0, low_seed_team: 0}
    rows = []
    game_date = context.current_date

    for game_number, home_slot in enumerate(SERIES_HOME_PATTERN, start=1):
        if wins[high_seed_team] == 4 or wins[low_seed_team] == 4:
            break

        game_date = game_date + pd.Timedelta(days=DEFAULT_PLAYOFF_REST_DAYS)
        home_team = high_seed_team if home_slot == "HIGH" else low_seed_team
        away_team = low_seed_team if home_slot == "HIGH" else high_seed_team
        winner, home_probability = simulate_single_game(context, model_bundle, game_date, home_team, away_team, rng)
        wins[winner] += 1
        rows.append(
            {
                "CONFERENCE": conference,
                "ROUND": round_name,
                "GAME_NUMBER": game_number,
                "HOME_TEAM": home_team,
                "AWAY_TEAM": away_team,
                "WINNER": winner,
                "HOME_WIN_PROB": home_probability,
                "SERIES_SCORE": f"{wins[high_seed_team]}-{wins[low_seed_team]}",
            }
        )

    context.current_date = game_date
    winner = high_seed_team if wins[high_seed_team] == 4 else low_seed_team
    return winner, rows


def simulate_conference_bracket(
    context: SimulationContext,
    model_bundle: dict,
    seeds: List[str],
    conference: str,
    rng: random.Random,
) -> Tuple[str, List[dict]]:
    seed_rank = {team: index + 1 for index, team in enumerate(seeds)}
    quarterfinal_pairings = [
        (seeds[0], seeds[7]),
        (seeds[3], seeds[4]),
        (seeds[1], seeds[6]),
        (seeds[2], seeds[5]),
    ]

    all_rows = []
    semifinalists = []
    for high_seed, low_seed in quarterfinal_pairings:
        winner, rows = simulate_series(context, model_bundle, high_seed, low_seed, "Conference Quarterfinals", conference, rng)
        semifinalists.append(winner)
        all_rows.extend(rows)

    semifinal_pairings = [
        (semifinalists[0], semifinalists[1]),
        (semifinalists[2], semifinalists[3]),
    ]
    finalists = []
    for high_seed, low_seed in semifinal_pairings:
        ordered = sorted([high_seed, low_seed], key=lambda team: seed_rank[team])
        winner, rows = simulate_series(context, model_bundle, ordered[0], ordered[1], "Conference Semifinals", conference, rng)
        finalists.append(winner)
        all_rows.extend(rows)

    ordered_finalists = sorted(finalists, key=lambda team: seed_rank[team])
    champion, rows = simulate_series(
        context,
        model_bundle,
        ordered_finalists[0],
        ordered_finalists[1],
        "Conference Finals",
        conference,
        rng,
    )
    all_rows.extend(rows)
    return champion, all_rows


def simulate_playoffs(
    context: SimulationContext,
    model_bundle: dict,
    east_seeds: List[str],
    west_seeds: List[str],
    rng: random.Random,
) -> Tuple[str, List[dict]]:
    east_champion, east_rows = simulate_conference_bracket(context, model_bundle, east_seeds, "East", rng)
    west_champion, west_rows = simulate_conference_bracket(context, model_bundle, west_seeds, "West", rng)

    finalists = sorted(
        [east_champion, west_champion],
        key=lambda team: (context.states[team].season_wins, context.states[team].elo),
        reverse=True,
    )
    nba_champion, finals_rows = simulate_series(context, model_bundle, finalists[0], finalists[1], "Finals", "NBA", rng)

    return nba_champion, east_rows + west_rows + finals_rows


def simulate_full_postseason(base_context: SimulationContext, model_bundle: dict, remaining_schedule: pd.DataFrame, rng_seed: int) -> dict:
    rng = random.Random(rng_seed)
    context = deepcopy(base_context)
    standings, regular_season_results = simulate_remaining_regular_season(context, remaining_schedule, model_bundle, rng)

    east_seeds, east_playin = simulate_playin_for_conference(context, model_bundle, standings, "East", rng)
    west_seeds, west_playin = simulate_playin_for_conference(context, model_bundle, standings, "West", rng)
    champion, playoff_rows = simulate_playoffs(context, model_bundle, east_seeds, west_seeds, rng)

    return {
        "champion": champion,
        "standings": standings,
        "regular_season_results": regular_season_results,
        "playin_results": pd.DataFrame(east_playin + west_playin),
        "playoff_results": pd.DataFrame(playoff_rows),
        "east_seeds": east_seeds,
        "west_seeds": west_seeds,
    }


def monte_carlo_summary(base_context: SimulationContext, model_bundle: dict, remaining_schedule: pd.DataFrame, runs: int) -> pd.DataFrame:
    champion_counts = defaultdict(int)
    finals_counts = defaultdict(int)
    conference_finals_counts = defaultdict(int)
    playoff_counts = defaultdict(int)
    playin_counts = defaultdict(int)

    latest_result = None
    for run_idx in range(runs):
        result = simulate_full_postseason(base_context, model_bundle, remaining_schedule, RANDOM_SEED + run_idx)
        latest_result = result
        champion_counts[result["champion"]] += 1

        east_seeds = result["east_seeds"]
        west_seeds = result["west_seeds"]
        for team in east_seeds + west_seeds:
            playoff_counts[team] += 1
        for team in east_seeds[6:8] + west_seeds[6:8]:
            playin_counts[team] += 1

        playoff_results = result["playoff_results"]
        conference_finalists = playoff_results[playoff_results["ROUND"] == "Conference Finals"]["HOME_TEAM"].tolist()
        conference_finalists += playoff_results[playoff_results["ROUND"] == "Conference Finals"]["AWAY_TEAM"].tolist()
        for team in set(conference_finalists):
            conference_finals_counts[team] += 1

        finals_teams = playoff_results[playoff_results["ROUND"] == "Finals"]["HOME_TEAM"].tolist()
        finals_teams += playoff_results[playoff_results["ROUND"] == "Finals"]["AWAY_TEAM"].tolist()
        for team in set(finals_teams):
            finals_counts[team] += 1

    all_teams = sorted(TEAM_CONFERENCES)
    summary = pd.DataFrame(
        {
            "TEAM": all_teams,
            "CONFERENCE": [TEAM_CONFERENCES[team] for team in all_teams],
            "PLAYOFF_PROB": [playoff_counts[team] / runs for team in all_teams],
            "PLAYIN_QUALIFIED_PROB": [playin_counts[team] / runs for team in all_teams],
            "CONF_FINALS_PROB": [conference_finals_counts[team] / runs for team in all_teams],
            "FINALS_PROB": [finals_counts[team] / runs for team in all_teams],
            "TITLE_PROB": [champion_counts[team] / runs for team in all_teams],
        }
    ).sort_values(["TITLE_PROB", "FINALS_PROB", "TEAM"], ascending=[False, False, True])

    if latest_result is not None:
        save_outputs(latest_result, summary)

    return summary.reset_index(drop=True)


def save_outputs(result: dict, summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    result["standings"].to_csv(OUTPUT_DIR / "standings_projection.csv", index=False)
    result["regular_season_results"].to_csv(OUTPUT_DIR / "regular_season_simulation.csv", index=False)
    result["playin_results"].to_csv(OUTPUT_DIR / "playin_results.csv", index=False)
    result["playoff_results"].to_csv(OUTPUT_DIR / "playoff_bracket.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "simulation_summary.csv", index=False)


def prompt_simulation_runs() -> int:
    raw_value = input(f"Numero de simulaciones Monte Carlo [{DEFAULT_SIMULATION_RUNS}]: ").strip()
    if not raw_value:
        return DEFAULT_SIMULATION_RUNS

    try:
        runs = int(raw_value)
    except ValueError as exc:
        raise ValueError("Debes introducir un numero entero de simulaciones.") from exc

    if runs <= 0:
        raise ValueError("El numero de simulaciones debe ser mayor que 0.")

    return runs


def main() -> None:
    simulation_runs = prompt_simulation_runs()
    model_bundle = load_model_bundle()
    base_context, latest_played_date = build_current_context(TARGET_SEASON)
    remaining_schedule = load_remaining_schedule(TARGET_SEASON, latest_played_date)

    print(f"Ultimo partido jugado en datos: {latest_played_date.date().isoformat()}")
    print(f"Partidos pendientes cargados: {len(remaining_schedule)}")
    print(f"Simulaciones Monte Carlo: {simulation_runs}")

    summary = monte_carlo_summary(base_context, model_bundle, remaining_schedule, simulation_runs)
    print("\nTop 10 probabilidades de titulo:")
    print(summary.head(10).to_string(index=False))
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
