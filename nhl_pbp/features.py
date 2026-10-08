"""NHL play-by-play -> laukausrivit xG-mallia varten.

Käyttää vain tietoja, jotka ovat saatavilla NHL:n api-web play-by-play -syötteestä, jotta sama
koodi toimii sekä koulutuksessa että tuotannossa (morning-hockey). Laukaus = shot-on-goal, missed-shot
tai goal (torjutut laukaukset pois, kuten MoneyPuckin datassa). Koordinaatit normalisoidaan niin,
että hyökkäysmaali on aina x=+89.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd

SHOT_EVENTS = {"shot-on-goal", "missed-shot", "goal"}
CTX_EVENTS = {"faceoff", "hit", "giveaway", "takeaway", "shot-on-goal", "missed-shot", "blocked-shot", "goal"}
GOAL_X = 89.0

CATEGORICAL = ["shotType", "lastEvent"]
NUMERIC = [
    "x", "absY", "dist", "angle", "period", "secInPeriod", "gameSec",
    "shooterSkaters", "defenderSkaters", "shooterGoalie", "emptyNet", "scoreDiff", "isHome",
    "sinceLast", "distLast", "speedLast", "lastSame", "lastX", "lastAbsY", "rebound", "rush",
    "sinceLastShotSame", "sinceLastShotOpp",
]
FEATURES = NUMERIC + CATEGORICAL


def _sec(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def _inferred_signs(plays):
    """Vanhoissa kausissa (ennen 2021-22) homeTeamDefendingSide puuttuu: päätellään hyökkäysmaalin
    puoli (x:n etumerkki) joukkueen hyökkäysalueen laukauksista per jakso."""
    xs = {}
    for p in plays:
        det = p.get("details", {})
        if p["typeDescKey"] in SHOT_EVENTS and det.get("zoneCode") == "O" and "xCoord" in det and "yCoord" in det:
            xs.setdefault((p["periodDescriptor"]["number"], det["eventOwnerTeamId"]), []).append(det["xCoord"])
    return {k: (1 if np.median(v) >= 0 else -1) for k, v in xs.items() if len(v) >= 2}


def game_shots(d):
    """Palauttaa listan laukaus-dikteja yhdestä pelistä (play-by-play JSON)."""
    home_id, away_id = d["homeTeam"]["id"], d["awayTeam"]["id"]
    plays = sorted(d["plays"], key=lambda p: p["sortOrder"])
    rows, score = [], {home_id: 0, away_id: 0}
    last = None                 # edellinen konteksti-tapahtuma
    last_shot = {}              # joukkue -> viimeisin laukausyritys (peliaika)
    signs = _inferred_signs(plays)
    for p in plays:
        typ = p["typeDescKey"]
        per = p["periodDescriptor"]
        if per["periodType"] == "SO" or per["number"] > 4:
            continue
        det = p.get("details", {})
        sec_in = _sec(p["timeInPeriod"])
        gsec = (per["number"] - 1) * 1200 + sec_in
        if typ in CTX_EVENTS and "xCoord" in det and "yCoord" in det and "eventOwnerTeamId" in det:
            team = det["eventOwnerTeamId"]
            home = team == home_id
            side = p.get("homeTeamDefendingSide")
            if side:
                flip = -1 if ((home and side == "right") or (not home and side == "left")) else 1
            else:
                flip = signs.get((per["number"], team))
            if flip:
                x, y = det["xCoord"] * flip, det["yCoord"] * flip
                if typ in SHOT_EVENTS:
                    sit = p.get("situationCode", "1551")
                    ag, as_, hs, hg = (int(c) for c in sit)
                    shooter_sk, def_sk = (hs, as_) if home else (as_, hs)
                    shooter_g, def_g = (hg, ag) if home else (ag, hg)
                    dist = math.hypot(GOAL_X - x, y)
                    ctx = {"lastEvent": "none", "sinceLast": 99.0, "distLast": 0.0, "speedLast": 0.0, "lastSame": -1,
                           "lastX": 0.0, "lastAbsY": 0.0, "rebound": 0, "rush": 0}
                    if last and last["per"] == per["number"]:
                        lx, ly = last["raw"]
                        lx, ly = lx * flip, ly * flip       # sama puoli kuin laukaus (sama jakso)
                        dt = max(gsec - last["gsec"], 0)
                        dl = math.hypot(x - lx, y - ly)
                        same = int(last["team"] == team)
                        ctx.update(lastEvent=last["typ"], sinceLast=float(dt), distLast=dl, speedLast=dl / max(dt, 1),
                                   lastSame=same, lastX=lx, lastAbsY=abs(ly),
                                   rebound=int(same and last["typ"] in ("shot-on-goal", "missed-shot") and dt <= 3),
                                   rush=int(dt <= 4 and lx < 25))
                    shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
                    rows.append({
                        "gameId": d["id"], "season": d["season"], "date": d["gameDate"], "eventId": p["eventId"],
                        "teamId": team, "oppId": away_id if home else home_id, "isHome": int(home),
                        "shooterId": shooter, "goalieId": det.get("goalieInNetId"),
                        "shotType": det.get("shotType", "unknown"),
                        "x": x, "absY": abs(y), "dist": dist,
                        "angle": math.degrees(math.atan2(abs(y), max(GOAL_X - x, 0.5))) if x < GOAL_X else 90.0,
                        "period": per["number"], "secInPeriod": sec_in, "gameSec": gsec,
                        "shooterSkaters": shooter_sk, "defenderSkaters": def_sk, "shooterGoalie": shooter_g,
                        "emptyNet": int(def_g == 0),
                        "scoreDiff": score[team] - score[away_id if home else home_id],
                        "sinceLastShotSame": float(gsec - last_shot[team]) if team in last_shot else 99.0,
                        "sinceLastShotOpp": float(gsec - last_shot[away_id if home else home_id]) if (away_id if home else home_id) in last_shot else 99.0,
                        "onGoal": int(typ != "missed-shot"), "goal": int(typ == "goal"),
                        **ctx,
                    })
                if typ in SHOT_EVENTS or typ == "blocked-shot":
                    last_shot[team] = gsec
                last = {"typ": typ, "team": team, "raw": (det["xCoord"], det["yCoord"]), "gsec": gsec, "per": per["number"]}
        if typ == "goal":
            score[home_id], score[away_id] = det.get("homeScore", score[home_id]), det.get("awayScore", score[away_id])
    return rows


def build(paths):
    rows = []
    for f in paths:
        rows += game_shots(json.loads(pathlib.Path(f).read_text()))
    df = pd.DataFrame(rows)
    df["shooterId"] = df["shooterId"].astype("Int64")
    df["goalieId"] = df["goalieId"].astype("Int64")
    return df


if __name__ == "__main__":
    import sys
    files = sorted(pathlib.Path("data/pbp").glob("*.json"))
    df = build(files)
    df.to_parquet("data/nhl_shots.parquet")
    print(len(files), "peliä,", len(df), "laukausta, maaliprosentti", round(df.goal.mean(), 4), "| on-goal sh%", round(df[df.onGoal == 1].goal.mean(), 4))
    print(df.groupby("season").size())
