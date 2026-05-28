def add_features(df):
    df = df.copy()

    df["elo_diff"] = df["HomeElo"] - df["AwayElo"]

    df["goal_diff"] = df["FTHG"] - df["FTAG"]

    df["shot_diff"] = df["HS"] - df["AS"]
    df["shots_ratio"] = df["HS"] / (df["AS"] + 1)

    df["corners_diff"] = df["HC"] - df["AC"]

    return df