import json
import os
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List

import pandas as pd


class EloRatings:
    def __init__(self, path: str = "models/elo.json"):
        self.path = path
        self.ratings = self.load()

    def load(self) -> Dict[str, float]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {str(k): float(v) for k, v in data.items()}
            except Exception:
                return {}
        return {}

    def save(self) -> None:
        os.makedirs("models", exist_ok=True)
        clean = {str(k): float(v) for k, v in self.ratings.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(clean, f)

    def get(self, team: str) -> float:
        team = str(team)
        if team not in self.ratings:
            self.ratings[team] = 1500.0
        return float(self.ratings[team])

    def update(self, home: str, away: str, result: str, k: int = 20) -> None:
        h = self.get(home)
        a = self.get(away)

        expected_home = 1 / (1 + 10 ** ((a - h) / 400))

        if result == "H":
            sh, sa = 1.0, 0.0
        elif result == "A":
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5

        self.ratings[str(home)] = h + k * (sh - expected_home)
        self.ratings[str(away)] = a + k * (sa - (1 - expected_home))