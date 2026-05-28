from dataclasses import dataclass, field
from collections import deque
from typing import Dict

import pandas as pd

from src.elo import EloRatings


FEATURE_COLUMNS = [
    "elo_diff",
    "form_diff",
    "home_form",
    "away_form",
    "attack_diff",
    "defense_diff",
    "shot_diff",
    "sot_diff",
    "corner_diff",
    "home_matches",
    "away_matches",
]


def _avg(values, default=0.0) -> float:
    return float(sum(values) / len(values)) if len(values) else float(default)


def _num(x, default=0.0) -> float:
    try:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


@dataclass
class TeamState:
    elo: float = 1500.0

    overall_points: deque = field(default_factory=lambda: deque(maxlen=5))
    overall_gf: deque = field(default_factory=lambda: deque(maxlen=5))
    overall_ga: deque = field(default_factory=lambda: deque(maxlen=5))

    home_points: deque = field(default_factory=lambda: deque(maxlen=5))
    away_points: deque = field(default_factory=lambda: deque(maxlen=5))

    home_gf: deque = field(default_factory=lambda: deque(maxlen=5))
    home_ga: deque = field(default_factory=lambda: deque(maxlen=5))
    away_gf: deque = field(default_factory=lambda: deque(maxlen=5))
    away_ga: deque = field(default_factory=lambda: deque(maxlen=5))

    home_shots_for: deque = field(default_factory=lambda: deque(maxlen=5))
    away_shots_for: deque = field(default_factory=lambda: deque(maxlen=5))

    home_sot_for: deque = field(default_factory=lambda: deque(maxlen=5))
    away_sot_for: deque = field(default_factory=lambda: deque(maxlen=5))

    home_corners_for: deque = field(default_factory=lambda: deque(maxlen=5))
    away_corners_for: deque = field(default_factory=lambda: deque(maxlen=5))

    home_matches: int = 0
    away_matches: int = 0


class LeagueTracker:
    def __init__(self):
        self.states: Dict[str, TeamState] = {}
        self.elo = EloRatings()

    def get_state(self, team: str) -> TeamState:
        team = str(team).strip()
        if team not in self.states:
            self.states[team] = TeamState(elo=self.elo.get(team))
        return self.states[team]

    def _result_points(self, result: str, is_home: bool) -> float:
        if result == "D":
            return 1.0
        if result == "H":
            return 3.0 if is_home else 0.0
        if result == "A":
            return 0.0 if is_home else 3.0
        return 0.0

    def make_features(self, home_team: str, away_team: str) -> dict:
        home = self.get_state(home_team)
        away = self.get_state(away_team)

        return {
            "elo_diff": float(home.elo - away.elo),
            "form_diff": _avg(home.overall_points) - _avg(away.overall_points),
            "home_form": _avg(home.home_points),
            "away_form": _avg(away.away_points),
            "attack_diff": _avg(home.home_gf) - _avg(away.away_gf),
            "defense_diff": _avg(away.away_ga) - _avg(home.home_ga),
            "shot_diff": _avg(home.home_shots_for) - _avg(away.away_shots_for),
            "sot_diff": _avg(home.home_sot_for) - _avg(away.away_sot_for),
            "corner_diff": _avg(home.home_corners_for) - _avg(away.away_corners_for),
            "home_matches": float(home.home_matches),
            "away_matches": float(away.away_matches),
        }

    def update_from_row(self, row: pd.Series) -> None:
        home_team = str(row.get("HomeTeam", "")).strip()
        away_team = str(row.get("AwayTeam", "")).strip()
        result = str(row.get("FTR", "")).strip()

        if not home_team or not away_team or not result or home_team == "nan" or away_team == "nan":
            return

        home = self.get_state(home_team)
        away = self.get_state(away_team)

        hg = _num(row.get("FTHG", 0))
        ag = _num(row.get("FTAG", 0))
        hs = _num(row.get("HS", 0))
        as_ = _num(row.get("AS", 0))
        hst = _num(row.get("HST", 0))
        ast = _num(row.get("AST", 0))
        hc = _num(row.get("HC", 0))
        ac = _num(row.get("AC", 0))

        home_points = self._result_points(result, is_home=True)
        away_points = self._result_points(result, is_home=False)

        # overall
        home.overall_points.append(home_points)
        away.overall_points.append(away_points)

        home.overall_gf.append(hg)
        home.overall_ga.append(ag)

        away.overall_gf.append(ag)
        away.overall_ga.append(hg)

        # home-specific
        home.home_points.append(home_points)
        home.home_gf.append(hg)
        home.home_ga.append(ag)
        home.home_shots_for.append(hs)
        home.home_sot_for.append(hst)
        home.home_corners_for.append(hc)
        home.home_matches += 1

        # away-specific
        away.away_points.append(away_points)
        away.away_gf.append(ag)
        away.away_ga.append(hg)
        away.away_shots_for.append(as_)
        away.away_sot_for.append(ast)
        away.away_corners_for.append(ac)
        away.away_matches += 1

        # Elo update AFTER features
        self.elo.update(home_team, away_team, result)
        home.elo = self.elo.get(home_team)
        away.elo = self.elo.get(away_team)


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, LeagueTracker]:
    tracker = LeagueTracker()
    rows = []

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        features = tracker.make_features(home_team, away_team)
        features["FTR"] = row["FTR"]
        rows.append(features)

        tracker.update_from_row(row)

    out = pd.DataFrame(rows)
    return out, tracker