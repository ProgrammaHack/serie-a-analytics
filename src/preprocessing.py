import pandas as pd


NUMERIC_COLS = [
    "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC",
    "B365H", "B365D", "B365A"
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # uniforma nomi
    df.columns = [c.strip() for c in df.columns]

    # pulizia stringhe squadra
    for col in ["HomeTeam", "AwayTeam", "FTR"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # parsing data robusto
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(
                df["Date"],
                dayfirst=True,
                errors="coerce",
                format="mixed"
            )
        except TypeError:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # conversione numerici
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # tieni solo righe buone
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    df = df.drop_duplicates()

    # ordina temporalmente
    sort_cols = ["Date"]
    if "Time" in df.columns:
        df["Time"] = df["Time"].astype("string").fillna("")
        sort_cols.append("Time")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df