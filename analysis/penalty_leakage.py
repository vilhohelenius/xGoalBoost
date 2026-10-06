"""Osoittaa, miksi *Penalty1TimeLeft/Length-sarakkeet vuotavat tuloksen.

Ylivoimamaali (5v4) päättää pienen rangaistuksen, joten maalin rivillä rangaistusaika on 0.
Ylivoimalla ja nollaan nollatulla rangaistusajalla maaliprosentti on ~68 % vs ~5 % muuten.
Ajo: .venv/bin/python analysis/penalty_leakage.py
"""
import glob
import numpy as np
import pandas as pd

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("shots_*.csv"))], ignore_index=True)
home = df.isHomeTeam == 1
shooter = np.where(home, df.homeSkatersOnIce, df.awaySkatersOnIce)
defender = np.where(home, df.awaySkatersOnIce, df.homeSkatersOnIce)
def_left = np.where(home, df.awayPenalty1TimeLeft, df.homePenalty1TimeLeft)
pp = df[(shooter == 5) & (defender == 4)].assign(defenderPenaltyLeft=def_left[(shooter == 5) & (defender == 4)])
print(pp.groupby(pd.cut(pp.defenderPenaltyLeft, [-1, 0, 5, 30, 60, 120])).goal.agg(["mean", "size"]))
