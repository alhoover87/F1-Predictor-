import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏁", layout="wide")
fastf1.Cache.enable_cache("/tmp")

QUALI_WEIGHTS = {"FP1": 0.20, "FP2": 0.35, "FP3": 0.45}
HISTORY_START_YEAR = 2021
GRID_WEIGHT_MIN = 0.15
GRID_WEIGHT_MAX = 0.55
GRID_WEIGHT_DEFAULT = 0.25


@st.cache_data(ttl=3600, show_spinner=False)
def get_schedule(year):
    return fastf1.get_event_schedule(year, include_testing=False)


def next_event(year):
    sched = get_schedule(year)
    date_col = "Session5DateUtc" if "Session5DateUtc" in sched.columns else "EventDate"
    dates = pd.to_datetime(sched[date_col], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    upcoming = sched[dates >= now]
    if len(upcoming) == 0:
        return sched.iloc[-1]
    return upcoming.iloc[0]


@st.cache_data(ttl=1800, show_spinner=False)
def load_session_laps(year, round_number, session_name):
    try:
        session = fastf1.get_session(year, round_number, session_name)
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()
        laps = laps[["Driver", "Team", "LapTime", "LapNumber", "Stint",
                     "Compound", "TyreLife", "IsAccurate", "PitInTime", "PitOutTime"]]
        laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
        return laps
    except Exception:
        return None


# ---------- Per-circuit history (2021 onward) ----------
@st.cache_data(ttl=86400, show_spinner=False)
def load_race_history(circuit_key, end_year):
    """Race results since HISTORY_START_YEAR for ONE circuit, matched on event Location."""
    rows = []
    for yr in range(HISTORY_START_YEAR, end_year + 1):
        try:
            sched = get_schedule(yr)
        except Exception:
            continue
        match = sched[sched["Location"].str.lower() == circuit_key.lower()]
        for _, ev in match.iterrows():
            try:
                sess = fastf1.get_session(yr, int(ev["RoundNumber"]), "Race")
                sess.load(laps=False, telemetry=False, weather=False, messages=False)
                res = sess.results[["Abbreviation", "GridPosition", "Position", "Status"]].copy()
                res["Year"] = yr
                rows.append(res)
            except Exception:
                continue
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={"Abbreviation": "Driver", "GridPosition": "Grid",
                            "Position": "Finish"})
    df["Grid"] = pd.to_numeric(df["Grid"], errors="coerce")
    df["Finish"] = pd.to_numeric(df["Finish"], errors="coerce")
    # pit-lane starters show as grid 0; treat as last on the grid
    df.loc[df["Grid"] == 0, "Grid"] = df["Grid"].max()
    # keep classified finishers only so crashes do not distort the averages
    df = df[df["Status"].astype(str).str.contains(r"Finished|\+\d+ Lap|Lapped", regex=True, na=False)]
    return df.dropna(subset=["Grid", "Finish"])


def grid_history_table(hist):
    if hist is None or hist.empty:
        return None
    t = hist.groupby("Grid").agg(
        Races=("Finish", "count"),
        AvgFinish=("Finish", "mean"),
        MedianFinish=("Finish", "median"),
    ).reset_index()
    t["PlacesGained"] = t["Grid"] - t["AvgFinish"]
    return t.round(2)


def circuit_grid_weight(hist):
    """Higher correlation between grid and finish = grid matters more here."""
    if hist is None or len(hist) < 40:
        return GRID_WEIGHT_DEFAULT, None
    corr = hist["Grid"].corr(hist["Finish"], method="spearman")
    if pd.isna(corr):
        return GRID_WEIGHT_DEFAULT, None
    # map correlation 0.3 -> min weight, 0.9 -> max weight
    scaled = (corr - 0.3) / (0.9 - 0.3)
    weight = GRID_WEIGHT_MIN + max(0.0, min(1.0, scaled)) * (GRID_WEIGHT_MAX - GRID_WEIGHT_MIN)
    return float(weight), float(corr)


def quali_projection(session_laps):
    rows = []
    for name, laps in session_laps.items():
        if laps is None or laps.empty:
            continue
        best = (laps.dropna(subset=["LapSeconds"])
                .groupby(["Driver", "Team"])["LapSeconds"].min().reset_index())
        fastest = best["LapSeconds"].min()
        best["GapPct"] = (best["LapSeconds"] - fastest) / fastest * 100
        best["Session"] = name
        rows.append(best)
    if not rows:
        return None
    allbest = pd.concat(rows)
    pivot = allbest.pivot_table(index=["Driver", "Team"], columns="Session",
                                values="GapPct").reset_index()
    score = pd.Series(0.0, index=pivot.index)
    total_w = pd.Series(0.0, index=pivot.index)
    for s, w in QUALI_WEIGHTS.items():
        if s in pivot.columns:
            has = pivot[s].notna()
            score[has] += pivot.loc[has, s] * w
            total_w[has] += w
    pivot["QualiScore"] = score / total_w.replace(0, np.nan)
    pivot = pivot.dropna(subset=["QualiScore"]).sort_values("QualiScore").reset_index(drop=True)
    pivot["ProjectedGrid"] = range(1, len(pivot) + 1)
    return pivot


def long_run_pace(session_laps):
    rows = []
    for name, laps in session_laps.items():
        if laps is None or laps.empty:
            continue
        clean = laps[(laps["IsAccurate"] == True) & laps["LapSeconds"].notna()
                     & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
                     & laps["TyreLife"].notna()]
        for (drv, team, stint), grp in clean.groupby(["Driver", "Team", "Stint"]):
            if len(grp) < 5:
                continue
            med = grp["LapSeconds"].median()
            grp = grp[grp["LapSeconds"] < med * 1.03]
            if len(grp) < 5:
                continue
            slope = np.polyfit(grp["TyreLife"].astype(float), grp["LapSeconds"], 1)[0]
            slope = max(min(slope, 0.2), -0.05)
            avg_age = grp["TyreLife"].mean()
            corrected = grp["LapSeconds"].mean() - slope * (avg_age - 5)
            rows.append({"Driver": drv, "Team": team, "Session": name,
                         "Laps": len(grp), "Compound": grp["Compound"].mode().iloc[0],
                         "Pace": corrected, "DegPerLap": slope})
    if not rows:
        return None
    runs = pd.DataFrame(rows)
    best_runs = runs.sort_values("Laps", ascending=False).groupby("Driver").head(2)
    agg = best_runs.groupby(["Driver", "Team"]).agg(
        RacePace=("Pace", "mean"), DegPerLap=("DegPerLap", "mean"),
        LongRunLaps=("Laps", "sum")).reset_index()
    fastest = agg["RacePace"].min()
    agg["RaceGapPct"] = (agg["RacePace"] - fastest) / fastest * 100
    return agg


def race_projection(quali, pace, grid_weight):
    if pace is None:
        return None
    df = pace.merge(quali[["Driver", "ProjectedGrid"]], on="Driver", how="left")
    df["ProjectedGrid"] = df["ProjectedGrid"].fillna(df["ProjectedGrid"].max() + 1)
    pace_rank = df["RaceGapPct"].rank()
    df["RaceScore"] = (1 - grid_weight) * pace_rank + grid_weight * df["ProjectedGrid"]
    df = df.sort_values("RaceScore").reset_index(drop=True)
    df["ProjectedFinish"] = df.index + 1
    return df


# ---------- UI ----------
st.title("🏁 F1 Podium Predictor")
this_year = datetime.now().year
year = int(st.sidebar.number_input("Season", 2021, this_year, this_year))
event = next_event(year)
round_number = int(event["RoundNumber"])
circuit = event["Location"]
st.subheader(f"{event['EventName']} – Round {round_number}")
st.caption(f"{circuit}, {event['Country']}")

sess_map = {"FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"}
session_laps = {}
with st.spinner("Loading practice data..."):
    for short, full in sess_map.items():
        session_laps[short] = load_session_laps(year, round_number, full)

available = [k for k, v in session_laps.items() if v is not None and not v.empty]
if not available:
    st.warning("No practice data is available yet for this weekend. Check back after FP1.")
    st.stop()
st.info("Practice sessions loaded: " + ", ".join(available))

with st.spinner(f"Loading {circuit} race history since {HISTORY_START_YEAR}..."):
    history = load_race_history(circuit, year - 1)
grid_weight, grid_corr = circuit_grid_weight(history)
hist_table = grid_history_table(history)

quali = quali_projection(session_laps)
pace = long_run_pace(session_laps)
race = race_projection(quali, pace, grid_weight) if quali is not None else None

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Podium", "⏱️ Qualifying", "🏎️ Race", "📈 History"])

with tab1:
    if race is None or race.empty:
        st.write("Not enough long-run data yet to project the race.")
    else:
        top3 = race.head(3)
        cols = st.columns(3)
        medals = ["🥇 P1", "🥈 P2", "🥉 P3"]
        for c, m, (_, r) in zip(cols, medals, top3.iterrows()):
            c.metric(m, r["Driver"], r["Team"], delta_color="off")
        st.caption(f"Grid position weight at {circuit}: {grid_weight:.0%}")

with tab2:
    if quali is None:
        st.write("No qualifying projection available.")
    else:
        show = quali[["ProjectedGrid", "Driver", "Team", "QualiScore"]].copy()
        show["QualiScore"] = show["QualiScore"].round(3)
        show = show.rename(columns={"QualiScore": "Avg gap to best (%)"})
        st.dataframe(show, hide_index=True, use_container_width=True)
        fig = px.bar(quali, x="Driver", y="QualiScore", color="Team",
                     title="Projected qualifying gap to fastest (%)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if race is None or race.empty:
        st.write("No race projection available yet.")
    else:
        show = race[["ProjectedFinish", "Driver", "Team", "ProjectedGrid",
                     "RaceGapPct", "DegPerLap", "LongRunLaps"]].copy()
        show["RaceGapPct"] = show["RaceGapPct"].round(3)
        show["DegPerLap"] = show["DegPerLap"].round(3)
        show = show.rename(columns={"RaceGapPct": "Long-run gap (%)",
                                    "DegPerLap": "Tyre deg (s/lap)",
                                    "LongRunLaps": "Long-run laps"})
        st.dataframe(show, hide_index=True, use_container_width=True)
        fig = px.scatter(race, x="RaceGapPct", y="DegPerLap", text="Driver",
                         color="Team", title="Long-run pace vs tyre degradation")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader(f"{circuit}: finishing position by grid slot, {HISTORY_START_YEAR} onward")
    if hist_table is None:
        st.write("No completed races at this circuit since 2021, so the default grid weight is used.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Races in sample", int(history["Year"].nunique()))
        c2.metric("Grid-to-finish correlation", f"{grid_corr:.2f}" if grid_corr is not None else "n/a")
        c3.metric("Grid weight used", f"{grid_weight:.0%}")
        shown = hist_table.rename(columns={
            "Grid": "Grid slot", "AvgFinish": "Avg finish",
            "MedianFinish": "Median finish", "PlacesGained": "Places gained"})
        st.dataframe(shown, hide_index=True, use_container_width=True)
        fig = px.line(hist_table, x="Grid", y="AvgFinish", markers=True,
                      title="Average finishing position by starting slot")
        fig.add_scatter(x=hist_table["Grid"], y=hist_table["Grid"], mode="lines",
                        name="No change", line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Classified finishers only. Points below the dashed line mean drivers from that slot tend to gain places.")

st.caption("Built from FastF1 data. Projections are estimates, not guarantees.")
