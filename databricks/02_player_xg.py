# Databricks notebook source
# MAGIC %md
# MAGIC # Pelaajan xG vs toteutuneet maalit
# MAGIC Lähde: `workspace.xgoalboost.skater_season_xg` (kaudet 2023-24...2025-26, runkosarja). Katkoviiva = maalit = xG.

# COMMAND ----------

import plotly.graph_objects as go
dbutils.widgets.dropdown("season", "20242025", ["20232024", "20242025", "20252026"])
dbutils.widgets.text("min_shots", "100")
season, min_shots = int(dbutils.widgets.get("season")), int(dbutils.widgets.get("min_shots"))
df = spark.table("workspace.xgoalboost.skater_season_xg").where(f"season = {season} AND shots >= {min_shots}").toPandas()
df["diff"] = df.goalsMinusXG

# COMMAND ----------

BLUE, ORANGE, GREY = "#2a6fbb", "#d9622b", "#9aa3ad"          # diverging: yli / alle odotuksen, harmaa = muut
top = df.nlargest(8, "diff"); bot = df.nsmallest(8, "diff")
fig = go.Figure()
rest = df.drop(top.index).drop(bot.index)
fig.add_scatter(x=rest.xG, y=rest.goals, mode="markers", marker=dict(color=GREY, size=7, opacity=.7), name="Muut", text=rest.name, hovertemplate="%{text}<br>xG %{x:.1f}, maalit %{y}")
fig.add_scatter(x=top.xG, y=top.goals, mode="markers+text", marker=dict(color=BLUE, size=9), text=top.name, textposition="top left", name="Yli xG", hovertemplate="%{text}<br>xG %{x:.1f}, maalit %{y}")
fig.add_scatter(x=bot.xG, y=bot.goals, mode="markers+text", marker=dict(color=ORANGE, size=9), text=bot.name, textposition="bottom right", name="Alle xG", hovertemplate="%{text}<br>xG %{x:.1f}, maalit %{y}")
m = float(max(df.xG.max(), df.goals.max())) + 2
fig.add_scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color=GREY, dash="dash", width=1), showlegend=False, hoverinfo="skip")
fig.update_layout(template="plotly_white", height=600, xaxis_title="Odotetut maalit (xG)", yaxis_title="Maalit", title=f"Pelaajat {season // 10000}-{season % 10000}, vähintään {min_shots} laukausta")
fig.show()

# COMMAND ----------

# MAGIC %md ## Eniten yli / alle odotuksen

# COMMAND ----------

display(spark.sql(f"""
SELECT name, goals, ROUND(xG,1) AS xG, ROUND(goals - xG,1) AS goals_over_xG, shots
FROM workspace.xgoalboost.skater_season_xg WHERE season = {season} AND shots >= {min_shots}
ORDER BY goals_over_xG DESC"""))

# COMMAND ----------

# MAGIC %md ## Maalivahdit: GSAx per 100 laukausta (vähintään 500 laukausta)

# COMMAND ----------

display(spark.sql(f"""
SELECT name, shotsAgainst, goalsAgainst, ROUND(xGA,1) AS xGA, ROUND(GSAx,1) AS GSAx, ROUND(GSAx_per_100,2) AS GSAx_per_100
FROM workspace.xgoalboost.goalie_season_xg WHERE season = {season} AND shotsAgainst >= 500
ORDER BY GSAx_per_100 DESC"""))
