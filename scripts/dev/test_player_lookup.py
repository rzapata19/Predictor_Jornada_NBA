import re

from nba_api.stats.endpoints import leaguedashplayerstats


CURRENT_SEASON = "2025-26"
TARGET_PLAYER = "Jonathan Kuminga"


def normalize_player_name(player_name: str) -> str:
    cleaned = str(player_name).strip()
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", cleaned).upper()
    cleaned = " ".join(cleaned.split())
    return cleaned


def main() -> None:
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=CURRENT_SEASON,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
    )
    df = stats.get_data_frames()[0].copy()

    print(f"Filas recibidas: {len(df)}")
    print("\nColumnas disponibles:")
    print(", ".join(df.columns.astype(str).tolist()))
    print("Primeros 5 jugadores:")
    preview_columns = [col for col in ["PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_ID", "MIN", "PTS", "GP"] if col in df.columns]
    print(df[preview_columns].head().to_string(index=False))

    df["PLAYER_KEY"] = df["PLAYER_NAME"].map(normalize_player_name)
    target_key = normalize_player_name(TARGET_PLAYER)

    exact_match = df[df["PLAYER_KEY"] == target_key]
    print(f"\nBuscando: {TARGET_PLAYER} -> {target_key}")

    if exact_match.empty:
        partial_match = df[df["PLAYER_KEY"].str.contains("KUMINGA", na=False)]
        print("\nNo hay match exacto. Coincidencias parciales por 'KUMINGA':")
        if partial_match.empty:
            print("Ninguna coincidencia parcial encontrada.")
        else:
            result_columns = [
                col for col in ["PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_ID", "MIN", "PTS", "GP", "PLAYER_KEY"] if col in partial_match.columns
            ]
            print(partial_match[result_columns].to_string(index=False))
        return

    print("\nMatch exacto encontrado:")
    result_columns = [col for col in ["PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_ID", "MIN", "PTS", "GP", "PLAYER_KEY"] if col in exact_match.columns]
    print(exact_match[result_columns].to_string(index=False))


if __name__ == "__main__":
    main()
