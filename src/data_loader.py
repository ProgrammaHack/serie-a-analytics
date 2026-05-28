import glob
import os
import pandas as pd


def load_data(path: str = "data/raw") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.csv")))

    if not files:
        raise FileNotFoundError(f"Nessun CSV trovato in: {path}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="latin1")
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out.columns = [c.strip() for c in out.columns]

    print(f"Loaded files: {len(files)}")
    print(f"Shape: {out.shape}")

    return out