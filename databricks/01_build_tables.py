# Databricks notebook source
# MAGIC %md
# MAGIC # Delta-taulut
# MAGIC Lataa valmiit tulokset Delta-tauluiksi (`workspace.xgoalboost.*`). Putken voi ajaa uudelleen alla (`RERUN = True`).
# MAGIC Skriptit lukevat datan ympäristömuuttujasta `XG_DATA`.

# COMMAND ----------

# MAGIC %pip install -q xgboost==3.4.1 pyarrow

# COMMAND ----------

import os, pathlib, subprocess, sys
ROOT = pathlib.Path(os.getcwd()).parent                  # repon juuri (notebook on kansiossa databricks/)
DATA = "/Volumes/workspace/xgoalboost/data/data"
os.environ["XG_DATA"] = DATA
RERUN = False                                            # True: laske xG ja ottelutaulut uudelleen play-by-playsta
if RERUN:
    for s in ("nhl_pbp/aggregate.py", "winprob/build_games.py"):
        subprocess.run([sys.executable, s], cwd=ROOT, check=True)

# COMMAND ----------

CSV = ["skater_season_xg", "skater_game_xg", "goalie_season_xg", "goalie_game_xg"]
PQ = ["nhl_shots", "wp_features", "wp_games", "wp_team_game", "wp_goalie_game"]
for t in CSV:
    spark.read.csv(f"{DATA}/{t}.csv", header=True, inferSchema=True).write.mode("overwrite").saveAsTable(f"workspace.xgoalboost.{t}")
for t in PQ:
    spark.read.parquet(f"{DATA}/{t}.parquet").write.mode("overwrite").saveAsTable(f"workspace.xgoalboost.{t}")
display(spark.sql("SHOW TABLES IN workspace.xgoalboost"))
