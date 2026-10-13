# Databricks notebook source
# MAGIC %md
# MAGIC # xGoalBoost: setup (Free Edition)
# MAGIC 1. Lisää repo: Workspace -> Create -> Git folder -> `https://github.com/vilhohelenius/xGoalBoost`
# MAGIC 2. Aja tämä notebook: luo skeeman ja Volumen. Lataa sitten `xgoalboost_data.tar.gz` Volumeen (Catalog -> workspace -> xgoalboost -> data -> Upload) ja aja loput solut.

# COMMAND ----------

CATALOG, SCHEMA, VOLUME = "workspace", "xgoalboost", "data"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
print(VOL)

# COMMAND ----------

# MAGIC %pip install -q xgboost==3.4.1 pyarrow plotly

# COMMAND ----------

import os, subprocess
VOL = "/Volumes/workspace/xgoalboost/data"
tar = f"{VOL}/xgoalboost_data.tar.gz"
if not os.path.exists(f"{VOL}/data/nhl_shots.parquet"):
    assert os.path.exists(tar), f"Lataa {tar} Volumeen ensin"
    subprocess.run(["tar", "-xzf", tar, "-C", VOL], check=True)    # -> {VOL}/data/...
print(os.listdir(f"{VOL}/data")[:20])
