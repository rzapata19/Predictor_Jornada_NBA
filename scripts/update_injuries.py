from pathlib import Path
import re
from io import StringIO
from difflib import get_close_matches
from typing import List, Optional

from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import teams as static_teams
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INJURIES_PATH = PROJECT_ROOT / "data" / "injuries_current.csv"
PLAYER_STATS_CACHE_PATH = PROJECT_ROOT / "data" / "player_stats_reference.csv"
ESPN_INJURIES_URL = "https://www.espn.com/nba/injuries"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/nba/",
    "Cache-Control": "no-cache",
}
ALL_TEAMS = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
]

TEAM_NAME_MAP = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}

STATUS_MAP = {
    "Day-To-Day": "Questionable",
    "Game Time Decision": "Questionable",
    "Out": "Out",
    "Doubtful": "Doubtful",
    "Questionable": "Questionable",
    "Probable": "Probable",
}
CURRENT_SEASON = "2025-26"
PLAYER_NAME_FIXES = {
    "Bones Hyland": "Nah'Shon Hyland",
    "Day'Ron Sharpe": "DayRon Sharpe",
    "Michael Porter Jr.": "Michael Porter Jr",
    "Craig Porter Jr.": "Craig Porter Jr",
    "Wendell Moore Jr.": "Wendell Moore Jr",
}


def normalize_team_name(team_name: str) -> str:
    cleaned = " ".join(str(team_name).split())
    return TEAM_NAME_MAP.get(cleaned, cleaned)


def normalize_status(status: str) -> str:
    cleaned = " ".join(str(status).split())
    return STATUS_MAP.get(cleaned, cleaned)


TEAM_ABBR_TO_NAME = {
    team["abbreviation"]: normalize_team_name(team["full_name"])
    for team in static_teams.get_teams()
}


def normalize_player_name(player_name: str) -> str:
    cleaned = str(player_name).strip()
    cleaned = PLAYER_NAME_FIXES.get(cleaned, cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", cleaned).upper()
    cleaned = " ".join(cleaned.split())
    return cleaned


def player_tokens(player_name: str) -> List[str]:
    return [token for token in normalize_player_name(player_name).split() if token]


def extract_team_names_from_html(html: str) -> List[str]:
    pattern_candidates = [
        r'alt="([^"]+)"',
        r'title="([^"]+)"',
        r'aria-label="([^"]+)"',
        r'data-team-name="([^"]+)"',
    ]
    ordered = []
    seen = set()

    for pattern in pattern_candidates:
        for match in re.findall(pattern, html, flags=re.IGNORECASE):
            team_name = normalize_team_name(match)
            if team_name in ALL_TEAMS and team_name not in seen:
                ordered.append(team_name)
                seen.add(team_name)

    return ordered


def extract_team_name_from_table(table: pd.DataFrame) -> Optional[str]:
    for column in table.columns:
        column_text = normalize_team_name(str(column))
        if column_text in ALL_TEAMS:
            return column_text

    for value in table.iloc[0].tolist() if not table.empty else []:
        value_text = normalize_team_name(str(value))
        if value_text in ALL_TEAMS:
            return value_text

    return None


def extract_team_tables() -> List[pd.DataFrame]:
    response = requests.get(ESPN_INJURIES_URL, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()
    html = response.text

    raw_tables = pd.read_html(StringIO(html))
    injury_tables: List[pd.DataFrame] = []
    for table in raw_tables:
        normalized_columns = [str(col).strip().upper() for col in table.columns]
        if "NAME" in normalized_columns or "PLAYER" in normalized_columns:
            table = table.copy()
            table.columns = normalized_columns
            injury_tables.append(table)

    team_names = extract_team_names_from_html(html)

    if len(team_names) < len(injury_tables):
        inferred_team_names = [extract_team_name_from_table(table) for table in injury_tables]
        merged_team_names = []
        html_team_iter = iter(team_names)

        for inferred_name in inferred_team_names:
            if inferred_name:
                merged_team_names.append(inferred_name)
            else:
                merged_team_names.append(next(html_team_iter, None))

        team_names = merged_team_names

    if not injury_tables:
        raise ValueError("No se encontraron tablas de lesiones en la pagina.")

    cleaned_tables: List[pd.DataFrame] = []
    skipped_tables = 0
    for team_name, table in zip(team_names, injury_tables):
        if not team_name:
            skipped_tables += 1
            continue
        table = table.copy()
        table["TEAM"] = team_name
        cleaned_tables.append(table)

    if not cleaned_tables:
        raise ValueError("No se pudo asociar ninguna tabla de lesiones a un equipo.")

    if skipped_tables:
        print(f"Aviso: se omitieron {skipped_tables} tablas sin equipo identificable.")

    return cleaned_tables


def fetch_player_reference() -> pd.DataFrame:
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=CURRENT_SEASON,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        )
        df = stats.get_data_frames()[0].copy()
        if df.empty:
            raise ValueError("LeagueDashPlayerStats devolvio un DataFrame vacio.")

        df["TEAM_NAME"] = df["TEAM_ABBREVIATION"].map(TEAM_ABBR_TO_NAME)
        df["PLAYER_KEY"] = df["PLAYER_NAME"].map(normalize_player_name)
        PLAYER_STATS_CACHE_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(PLAYER_STATS_CACHE_PATH, index=False)
        print(f"Referencia de jugadores actualizada: {PLAYER_STATS_CACHE_PATH}")
        return df
    except Exception as exc:
        if PLAYER_STATS_CACHE_PATH.exists():
            print(f"Aviso: fallo al descargar stats de jugadores. Se usara cache local. Motivo: {exc}")
            cached_df = pd.read_csv(PLAYER_STATS_CACHE_PATH)
            if "TEAM_NAME" not in cached_df.columns and "TEAM_ABBREVIATION" in cached_df.columns:
                cached_df["TEAM_NAME"] = cached_df["TEAM_ABBREVIATION"].map(TEAM_ABBR_TO_NAME)
            cached_df["TEAM_NAME"] = cached_df["TEAM_NAME"].map(normalize_team_name)
            cached_df["PLAYER_KEY"] = cached_df["PLAYER_NAME"].map(normalize_player_name)
            return cached_df
        raise RuntimeError(
            "No se pudieron cargar stats de jugadores ni existe cache local. "
            "Ejecuta de nuevo mas tarde o revisa el acceso a nba_api."
        ) from exc


def infer_player_stats(player_name: str, team_name: str, player_reference: pd.DataFrame) -> tuple[float, float, int, str]:
    if player_reference.empty:
        return 0.0, 0.0, 0, normalize_team_name(team_name)

    player_key = normalize_player_name(player_name)
    canonical_team_name = normalize_team_name(team_name)
    exact_match = player_reference[
        (player_reference["PLAYER_KEY"] == player_key)
        & (player_reference["TEAM_NAME"] == canonical_team_name)
    ]

    if exact_match.empty:
        exact_match = player_reference[player_reference["PLAYER_KEY"] == player_key]

    candidate_df = exact_match
    if candidate_df.empty:
        same_team_df = player_reference[player_reference["TEAM_NAME"] == canonical_team_name].copy()
        if not same_team_df.empty:
            matches = get_close_matches(player_key, same_team_df["PLAYER_KEY"].tolist(), n=1, cutoff=0.72)
            if matches:
                candidate_df = same_team_df[same_team_df["PLAYER_KEY"] == matches[0]]

    if candidate_df.empty:
        matches = get_close_matches(player_key, player_reference["PLAYER_KEY"].tolist(), n=1, cutoff=0.85)
        if matches:
            candidate_df = player_reference[player_reference["PLAYER_KEY"] == matches[0]]

    if candidate_df.empty:
        tokens = player_tokens(player_name)
        if len(tokens) >= 2:
            last_token = tokens[-1]
            first_token = tokens[0]
            token_match_df = player_reference[
                player_reference["PLAYER_KEY"].str.contains(last_token, na=False)
                & player_reference["PLAYER_KEY"].str.contains(first_token, na=False)
            ]
            if token_match_df.empty and len(tokens) >= 3:
                middle_token = tokens[1]
                token_match_df = player_reference[
                    player_reference["PLAYER_KEY"].str.contains(last_token, na=False)
                    & player_reference["PLAYER_KEY"].str.contains(middle_token, na=False)
                ]
            if not token_match_df.empty:
                same_team_match_df = token_match_df[token_match_df["TEAM_NAME"] == canonical_team_name]
                candidate_df = same_team_match_df if not same_team_match_df.empty else token_match_df

    if candidate_df.empty:
        return 0.0, 0.0, 0, canonical_team_name

    if len(candidate_df) > 1:
        team_match_df = candidate_df[candidate_df["TEAM_NAME"] == canonical_team_name]
        if not team_match_df.empty:
            candidate_df = team_match_df

    record = candidate_df.iloc[0]
    minutes_per_game = float(record["MIN"]) if "MIN" in record else 0.0
    points_per_game = float(record["PTS"]) if "PTS" in record else 0.0
    games_played = float(record["GP"]) if "GP" in record and float(record["GP"]) > 0 else 0.0
    is_starter = int(minutes_per_game >= 24.0 and games_played >= 5)
    resolved_team_name = normalize_team_name(record["TEAM_NAME"]) if "TEAM_NAME" in record else canonical_team_name

    return minutes_per_game, points_per_game, is_starter, resolved_team_name


def build_injuries_dataframe() -> pd.DataFrame:
    team_tables = extract_team_tables()
    try:
        player_reference = fetch_player_reference()
    except Exception as exc:
        print(f"Aviso: no se pudieron cargar stats de jugadores. Se usaran valores neutros. Motivo: {exc}")
        player_reference = pd.DataFrame()
    rows = []
    resolved_stats = 0
    unresolved_stats = []

    for table in team_tables:
        for _, row in table.iterrows():
            player_name = row["NAME"] if "NAME" in table.columns else row.get("PLAYER", "")
            status = normalize_status(row.get("STATUS", ""))
            return_estimate = row.get("EST. RETURN DATE", row.get("EST_RETURN_DATE", ""))
            comment = row.get("COMMENT", "")

            if not str(player_name).strip():
                continue

            minutes_per_game, points_per_game, is_starter, resolved_team_name = infer_player_stats(
                str(player_name).strip(),
                normalize_team_name(row["TEAM"]),
                player_reference,
            )
            if minutes_per_game > 0 or points_per_game > 0 or is_starter > 0 or resolved_team_name != normalize_team_name(row["TEAM"]):
                resolved_stats += 1
            else:
                unresolved_stats.append(f"{normalize_team_name(row['TEAM'])} | {str(player_name).strip()}")
            rows.append(
                {
                    "TEAM": resolved_team_name,
                    "PLAYER": str(player_name).strip(),
                    "STATUS": status,
                    "MINUTES_PER_GAME": minutes_per_game,
                    "POINTS_PER_GAME": points_per_game,
                    "IS_STARTER": is_starter,
                    "RETURN_ESTIMATE": str(return_estimate).strip(),
                    "NOTES": str(comment).strip(),
                }
            )

    if not rows:
        raise ValueError("No se pudieron extraer lesiones desde ESPN. Revisa si la estructura de la pagina ha cambiado.")

    injuries_df = pd.DataFrame(rows)
    injuries_df = injuries_df[injuries_df["STATUS"].isin({"Out", "Doubtful", "Questionable", "Probable"})].copy()
    injuries_df = injuries_df.sort_values(["TEAM", "STATUS", "PLAYER"]).reset_index(drop=True)
    print(f"Jugadores con stats resueltas: {resolved_stats}/{len(injuries_df)}")
    if unresolved_stats:
        print("Primeros jugadores sin resolver:")
        for item in unresolved_stats[:10]:
            print(f"- {item}")
    return injuries_df


def main() -> None:
    injuries_df = build_injuries_dataframe()
    INJURIES_PATH.parent.mkdir(exist_ok=True)
    injuries_df.to_csv(INJURIES_PATH, index=False)

    print(f"Lesiones actualizadas en: {INJURIES_PATH}")
    print(f"Registros guardados: {len(injuries_df)}")
    print(injuries_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
