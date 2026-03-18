from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INJURIES_PATH = PROJECT_ROOT / "data" / "injuries_current.csv"

ACTIVE_STATUSES = {"Out", "Doubtful", "Questionable"}


def main() -> None:
    if not INJURIES_PATH.exists():
        raise FileNotFoundError(f"No existe el fichero de lesiones: {INJURIES_PATH}")

    df = pd.read_csv(INJURIES_PATH)
    required_columns = {
        "TEAM",
        "PLAYER",
        "STATUS",
        "MINUTES_PER_GAME",
        "POINTS_PER_GAME",
        "IS_STARTER",
        "RETURN_ESTIMATE",
        "NOTES",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en injuries_current.csv: {sorted(missing)}")

    df = df[df["STATUS"].isin(ACTIVE_STATUSES)].copy()

    if df.empty:
        print("No hay lesiones activas registradas.")
        return

    df["IS_STARTER"] = df["IS_STARTER"].astype(int)
    df["MINUTES_PER_GAME"] = df["MINUTES_PER_GAME"].astype(float)
    df["POINTS_PER_GAME"] = df["POINTS_PER_GAME"].astype(float)

    team_summary = (
        df.groupby("TEAM", as_index=False)
        .agg(
            PLAYERS_OUT=("PLAYER", "count"),
            MINUTES_OUT=("MINUTES_PER_GAME", "sum"),
            POINTS_OUT=("POINTS_PER_GAME", "sum"),
            STARTERS_OUT=("IS_STARTER", "sum"),
        )
        .sort_values(["MINUTES_OUT", "POINTS_OUT", "TEAM"], ascending=[False, False, True])
    )

    print("Resumen de lesiones activas")
    print("-" * 90)
    print(team_summary.to_string(index=False))
    print("\nDetalle por jugador")
    print("-" * 90)
    print(
        df.sort_values(["TEAM", "STATUS", "MINUTES_PER_GAME"], ascending=[True, True, False])[
            ["TEAM", "PLAYER", "STATUS", "MINUTES_PER_GAME", "POINTS_PER_GAME", "IS_STARTER", "RETURN_ESTIMATE", "NOTES"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
