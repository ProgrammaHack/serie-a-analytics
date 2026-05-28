import requests

seasons = [
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425"
]

base_url = "https://www.football-data.co.uk/mmz4281/"

for season in seasons:
    url = f"{base_url}{season}/I1.csv"

    response = requests.get(url)

    with open(f"serie_a_{season}.csv", "wb") as f:
        f.write(response.content)

    print(f"Downloaded {season}")