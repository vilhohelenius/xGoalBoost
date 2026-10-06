"""Lataa NHL:n play-by-play (runkosarja) levyvälimuistiin data/pbp/<gameId>.json.

Ajo: python nhl_pbp/download.py 2023 2024 2025
Vain pelatut ottelut (gameState OFF/FINAL) tallennetaan; puuttuvat id:t ohitetaan.
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = pathlib.Path("data/pbp")
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"


def fetch(gid):
    f = OUT / f"{gid}.json"
    if f.exists():
        return gid, "cached"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL.format(gid), headers={"User-Agent": "xgoalboost"}), timeout=30) as r:
                d = json.load(r)
            if d.get("gameState") not in ("OFF", "FINAL"):
                return gid, "unfinished"
            f.write_text(json.dumps(d, separators=(",", ":")))
            return gid, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return gid, "404"
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return gid, "failed"


if __name__ == "__main__":
    ids = [int(f"{s}02{n:04d}") for s in map(int, sys.argv[1:]) for n in range(1, 1400)]
    stats = {}
    with ThreadPoolExecutor(6) as ex:
        for i, (gid, st) in enumerate(ex.map(fetch, ids)):
            stats[st] = stats.get(st, 0) + 1
            if i % 500 == 0:
                print(i, stats, flush=True)
    print("valmis", stats)
