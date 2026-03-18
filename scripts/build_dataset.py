from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Deque, Dict, List, Optional, Set, Tuple

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

BASE_ELO = 1500
BASE_K = 24
HOME_ADV = 75
ROLLING_WINDOWS = (3, 5, 10)
MIN_REST_DAYS = 0
MAX_REST_DAYS = 5
SEASON_REVERT = 0.75
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_ml.csv"

seasons = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

team_map = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}


@dataclass
class TeamState:
    season_games: int = 0
    season_wins: int = 0
    season_points_for: int = 0
    season_points_against: int = 0
    season_home_games: int = 0
    season_home_wins: int = 0
    season_away_games: int = 0
    season_away_wins: int = 0
    elo: float = BASE_ELO
    last_date: Optional[pd.Timestamp] = None
    last_home_date: Optional[pd.Timestamp] = None
    last_away_date: Optional[pd.Timestamp] = None
    results: Deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    margin: Deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    points_for: Deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    points_against: Deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    home_results: Deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))
    away_results: Deque[int] = field(default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS)))


def expected_result(home_elo: float, away_elo: float) -> float:
    return 1 / (1 + 10 ** (((away_elo) - (home_elo + HOME_ADV)) / 400))


def margin_multiplier(home_points: int, away_points: int, elo_diff: float) -> float:
    margin = abs(home_points - away_points)
    return ((margin + 3) ** 0.8) / (7.5 + 0.006 * abs(elo_diff))


def update_elo(home_elo: float, away_elo: float, home_win: int, home_points: int, away_points: int) -> Tuple[float, float]:
    exp_home = expected_result(home_elo, away_elo)
    elo_diff = home_elo - away_elo
    k = BASE_K * margin_multiplier(home_points, away_points, elo_diff)
    new_home = home_elo + k * (home_win - exp_home)
    new_away = away_elo + k * ((1 - home_win) - (1 - exp_home))
    return new_home, new_away


def safe_mean(values: deque, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def window_mean(values: deque, window: int, default: float = 0.0) -> float:
    if not values:
        return default
    recent = list(values)[-window:]
    return sum(recent) / len(recent)


def capped_rest(current_date: pd.Timestamp, previous_date: Optional[pd.Timestamp]) -> int:
    if previous_date is None:
        return 3
    rest_days = (current_date - previous_date).days
    return max(MIN_REST_DAYS, min(MAX_REST_DAYS, rest_days))


def season_regressed_elo(previous_states: Dict[str, TeamState], teams_in_season: Set[str]) -> Dict[str, TeamState]:
    states: Dict[str, TeamState] = {}
    for team in teams_in_season:
        prior_elo = previous_states.get(team).elo if team in previous_states else BASE_ELO
        regressed_elo = BASE_ELO + (prior_elo - BASE_ELO) * SEASON_REVERT
        states[team] = TeamState(elo=regressed_elo)
    return states


def season_sort_key(season: str) -> Tuple[int, int]:
    start_year, end_year = season.split("-")
    return int(start_year), int(end_year)


def copy_elo_states(states: Dict[str, TeamState]) -> Dict[str, TeamState]:
    return {team: TeamState(elo=state.elo) for team, state in states.items()}


def ensure_team_state(team: str, active_states: Dict[str, TeamState], previous_states: Dict[str, TeamState]) -> TeamState:
    if team not in active_states:
        prior_elo = previous_states.get(team).elo if team in previous_states else BASE_ELO
        regressed_elo = BASE_ELO + (prior_elo - BASE_ELO) * SEASON_REVERT
        active_states[team] = TeamState(elo=regressed_elo)
    return active_states[team]


def download_games(target_seasons: List[str]) -> pd.DataFrame:
    frames = []

    for season in target_seasons:
        print("Downloading games", season)

        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
        )

        df = finder.get_data_frames()[0]
        df = df[df["MATCHUP"].notna()]

        home = df[df["MATCHUP"].str.contains("vs.")]
        away = df[df["MATCHUP"].str.contains("@")]

        merged = pd.merge(
            home,
            away,
            on="GAME_ID",
            suffixes=("_home", "_away"),
        )

        games = pd.DataFrame(
            {
                "GAME_ID": merged["GAME_ID"],
                "DATE": pd.to_datetime(merged["GAME_DATE_home"]),
                "HOME_TEAM": merged["TEAM_NAME_home"].replace(team_map),
                "AWAY_TEAM": merged["TEAM_NAME_away"].replace(team_map),
                "HOME_POINTS": merged["PTS_home"].astype(int),
                "AWAY_POINTS": merged["PTS_away"].astype(int),
                "SEASON": season,
            }
        )

        frames.append(games)
        time.sleep(1.5)

    if not frames:
        return pd.DataFrame(columns=["GAME_ID", "DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_POINTS", "AWAY_POINTS", "SEASON"])

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.drop_duplicates(subset=["GAME_ID"])
    dataset = dataset.sort_values(["DATE", "GAME_ID"]).reset_index(drop=True)
    return dataset


def load_existing_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATASET_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.sort_values(["DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)


def add_window_features(row: dict, prefix: str, state: TeamState) -> None:
    for window in ROLLING_WINDOWS:
        row[f"{prefix}_WINRATE_LAST_{window}"] = window_mean(state.results, window, 0.5)
        row[f"{prefix}_MARGIN_LAST_{window}"] = window_mean(state.margin, window, 0.0)
        row[f"{prefix}_POINTS_FOR_LAST_{window}"] = window_mean(state.points_for, window, 110.0)
        row[f"{prefix}_POINTS_AGAINST_LAST_{window}"] = window_mean(state.points_against, window, 110.0)


def build_row(game: pd.Series, home_state: TeamState, away_state: TeamState, head_to_head: defaultdict) -> dict:
    home = game["HOME_TEAM"]
    away = game["AWAY_TEAM"]
    date = game["DATE"]

    home_rest = capped_rest(date, home_state.last_date)
    away_rest = capped_rest(date, away_state.last_date)
    home_site_rest = capped_rest(date, home_state.last_home_date)
    away_site_rest = capped_rest(date, away_state.last_away_date)

    h2h_key = tuple(sorted((home, away)))
    h2h_history = head_to_head[h2h_key]
    home_h2h_winrate = safe_mean(
        deque([result if first_team == home else 1 - result for first_team, result in h2h_history], maxlen=5),
        0.5,
    )

    row = {
        "DATE": date,
        "SEASON": game["SEASON"],
        "HOME_TEAM": home,
        "AWAY_TEAM": away,
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
        "HOME_WIN": int(game["HOME_POINTS"] > game["AWAY_POINTS"]),
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


def apply_result_to_state(game: pd.Series, home_state: TeamState, away_state: TeamState, head_to_head: defaultdict) -> None:
    home = game["HOME_TEAM"]
    away = game["AWAY_TEAM"]
    date = game["DATE"]
    home_points = int(game["HOME_POINTS"])
    away_points = int(game["AWAY_POINTS"])
    home_win = int(home_points > away_points)

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

    h2h_key = tuple(sorted((home, away)))
    head_to_head[h2h_key].append((home, home_win))


def synthetic_game_from_dataset_row(row: pd.Series) -> pd.Series:
    probability_home = float(row["EXPECTED_HOME_WIN"]) if "EXPECTED_HOME_WIN" in row else 0.5
    margin = int(round((probability_home - 0.5) * 24))
    margin = max(1, abs(margin))
    home_win = int(row["HOME_WIN"])
    home_points = 110 + margin if home_win else 110 - margin
    away_points = 110 - margin if home_win else 110 + margin

    return pd.Series(
        {
            "DATE": pd.Timestamp(row["DATE"]),
            "SEASON": row["SEASON"],
            "HOME_TEAM": row["HOME_TEAM"],
            "AWAY_TEAM": row["AWAY_TEAM"],
            "HOME_POINTS": home_points,
            "AWAY_POINTS": away_points,
        }
    )


def reconstruct_state_from_dataset(existing_df: pd.DataFrame) -> Tuple[str, Dict[str, TeamState], Dict[str, TeamState], defaultdict]:
    previous_states: Dict[str, TeamState] = {}
    head_to_head: defaultdict[Tuple[str, str], Deque[Tuple[str, int]]] = defaultdict(lambda: deque(maxlen=5))
    season_states: Dict[str, Dict[str, TeamState]] = {}

    season_team_sets = {
        season: set(
            pd.concat(
                [
                    existing_df.loc[existing_df["SEASON"] == season, "HOME_TEAM"],
                    existing_df.loc[existing_df["SEASON"] == season, "AWAY_TEAM"],
                ]
            ).unique()
        )
        for season in sorted(existing_df["SEASON"].unique(), key=season_sort_key)
    }

    for season in sorted(season_team_sets, key=season_sort_key):
        active_states = season_regressed_elo(previous_states, season_team_sets[season])
        season_states[season] = active_states
        season_rows = existing_df[existing_df["SEASON"] == season].sort_values(["DATE", "HOME_TEAM", "AWAY_TEAM"])

        for _, row in season_rows.iterrows():
            game = synthetic_game_from_dataset_row(row)
            home_state = active_states[game["HOME_TEAM"]]
            away_state = active_states[game["AWAY_TEAM"]]
            apply_result_to_state(game, home_state, away_state, head_to_head)

        previous_states = copy_elo_states(active_states)

    last_season = max(existing_df["SEASON"].unique(), key=season_sort_key)
    return last_season, season_states[last_season], previous_states, head_to_head


def select_seasons_to_download(existing_df: pd.DataFrame) -> List[str]:
    if existing_df.empty:
        return seasons

    last_season = max(existing_df["SEASON"].unique(), key=season_sort_key)
    return [season for season in seasons if season_sort_key(season) >= season_sort_key(last_season)]


def build_dataset() -> None:
    existing_df = load_existing_dataset()
    target_seasons = select_seasons_to_download(existing_df)
    downloaded_games = download_games(target_seasons)

    if existing_df.empty:
        games_to_process = downloaded_games
        previous_states: Dict[str, TeamState] = {}
        active_states: Dict[str, TeamState] = {}
        active_season = None
        head_to_head: defaultdict[Tuple[str, str], Deque[Tuple[str, int]]] = defaultdict(lambda: deque(maxlen=5))
        new_rows: List[dict] = []
    else:
        last_date = existing_df["DATE"].max()
        games_to_process = downloaded_games[downloaded_games["DATE"] > last_date].copy()
        if games_to_process.empty:
            print(f"No hay partidos nuevos desde {last_date.date().isoformat()}. Dataset sin cambios.")
            return

        active_season, active_states, previous_states, head_to_head = reconstruct_state_from_dataset(existing_df)
        new_rows = []
        print(f"Actualizacion incremental desde {last_date.date().isoformat()} con {len(games_to_process)} partidos nuevos.")

    games_to_process = games_to_process.sort_values(["DATE", "GAME_ID"]).reset_index(drop=True)

    for _, game in games_to_process.iterrows():
        game_season = game["SEASON"]

        if active_season != game_season:
            previous_states = copy_elo_states(active_states) if active_states else previous_states
            active_states = {}
            active_season = game_season

        home_state = ensure_team_state(game["HOME_TEAM"], active_states, previous_states)
        away_state = ensure_team_state(game["AWAY_TEAM"], active_states, previous_states)

        row = build_row(game, home_state, away_state, head_to_head)
        new_rows.append(row)
        apply_result_to_state(game, home_state, away_state, head_to_head)

    new_df = pd.DataFrame(new_rows)
    if existing_df.empty:
        final_df = new_df
        print("Dataset creado:", final_df.shape)
    else:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM"])
        final_df = final_df.sort_values(["DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM"]).reset_index(drop=True)
        print("Dataset actualizado:", final_df.shape)

    DATASET_PATH.parent.mkdir(exist_ok=True)
    final_df.to_csv(DATASET_PATH, index=False)
    print(f"Guardado en: {DATASET_PATH}")


if __name__ == "__main__":
    build_dataset()
