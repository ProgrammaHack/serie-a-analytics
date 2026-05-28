import pandas as pd
import glob

files = glob.glob("*.csv")

dfs = [pd.read_csv(f) for f in files]

matches = pd.concat(dfs)

print(matches.head())
print(matches.columns.tolist())