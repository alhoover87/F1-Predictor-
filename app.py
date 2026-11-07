import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import logging
import time
from datetime import datetime

st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏁", layout="wide")
fastf1.Cache.enable_cache("/tmp")

QUALI_WEIGHTS = {"FP1": 0.20, "FP2": 0.35, "FP3": 0.45}
FIRST_SEASON = 2021
GRID_WEIGHT_DEFAULT = 25  # percent; adjustable in the sidebar


@st.cache_data(ttl=3600, show_spinner=False)
def get_schedule(year):
    return fastf1.get_event_schedule(year, include_testing=False)


def selectable_events(year):
    """Every round that has started, plus the next upcoming one.
    Returns (schedule rows, index of the default pick)."""
    sched = get_schedule(year)
    sched = sched[sched["RoundNumber"] > 0].reset_index(drop=True)
    start_col = "Session1DateUtc" if "Session1DateUtc" in sched.columns else "EventDate"
    end_col = "Session5DateUtc" if "Session5DateUtc" in sched.columns else "EventDate"
    starts = pd.to_datetime(sched[start_col], utc=True, errors="coerce")
    ends = pd.to_datetime(sched[end_col], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    upcoming = sched.index[ends >= now]
    if len(upcoming) == 0:          # season over: show all, default to the finale
        return sched, len(sched) - 1
    nxt = upcoming[0]
    started = sched.index[starts <= now]
    last = max(nxt, started.max() if len(started) else nxt)
    return sched.loc[:last], int(nxt)


def next_event(year):
    sched = get_schedule(year)
    date_col = "Session5DateUtc" if "Session5DateUtc" in sched.columns else "EventDate"
    dates = pd.to_datetime(sched[date_col], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    upcoming = sched[dates >= now]
    if len(upcoming) == 0:
        return sched.iloc[-1]
    return upcoming.iloc[0]


LAP_COLUMNS = ["Driver", "Team", "LapTime", "LapNumber", "Stint", "Compound",
               "TyreLife", "IsAccurate", "PitInTime", "PitOutTime", "LapSeconds"]


class _LogCapture(logging.Handler):
    """Collects FastF1's warning/error log lines, which it logs instead of raising."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@st.cache_data(ttl=1800, show_spinner=False)
def _load_fastf1_laps(year, round_number, session_name):
    # Raises on failure, so failures are NOT cached and get retried next time
    cap = _LogCapture()
    logger = logging.getLogger("fastf1")
    logger.addHandler(cap)
    try:
        session = fastf1.get_session(year, round_number, session_name)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        try:
            laps = session.laps.copy()
        except Exception as e:
            detail = " | ".join(cap.lines[-3:]) or str(e)
            raise RuntimeError(f"FastF1 could not get timing data ({detail})")
    finally:
        logger.removeHandler(cap)
    if laps is None or laps.empty:
        raise ValueError("FastF1 loaded the session but it has no laps yet")
    laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
    return laps[LAP_COLUMNS]


OPENF1 = "https://api.openf1.org/v1"


def _openf1_get(endpoint, **params):
    for attempt in range(3):
        r = requests.get(f"{OPENF1}/{endpoint}", params=params, timeout=30)
        if r.status_code == 429:  # rate limited, back off and retry
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"OpenF1 rate limit on {endpoint}")


@st.cache_data(ttl=1800, show_spinner=False)
def _load_openf1_laps(year, session_name, session_date_utc):
    """Backup source. Finds the session by name and date, then builds the same
    lap table FastF1 would give us."""
    sessions = _openf1_get("sessions", year=year, session_name=session_name)
    if not sessions:
        raise ValueError(f"OpenF1 has no '{session_name}' sessions for {year}")
    target = pd.Timestamp(session_date_utc)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    best = min(sessions, key=lambda x: abs(pd.Timestamp(x["date_start"]).tz_convert("UTC") - target))
    if abs(pd.Timestamp(best["date_start"]).tz_convert("UTC") - target) > pd.Timedelta(days=2):
        raise ValueError("OpenF1 has not published this session yet")
    key = best["session_key"]

    laps = pd.DataFrame(_openf1_get("laps", session_key=key))
    time.sleep(0.4)
    stints = pd.DataFrame(_openf1_get("stints", session_key=key))
    time.sleep(0.4)
    drivers = pd.DataFrame(_openf1_get("drivers", session_key=key))
    if laps.empty:
        raise ValueError("OpenF1 found the session but it has no laps yet")

    drivers = drivers.drop_duplicates("driver_number")[["driver_number", "name_acronym", "team_name"]]
    laps = laps.merge(drivers, on="driver_number", how="left")

    laps["Stint"] = np.nan
    laps["Compound"] = None
    laps["TyreLife"] = np.nan
    laps["PitInTime"] = np.nan
    if not stints.empty:
        for _, stn in stints.iterrows():
            if pd.isna(stn.get("lap_start")) or pd.isna(stn.get("lap_end")):
                continue
            m = ((laps["driver_number"] == stn["driver_number"])
                 & (laps["lap_number"] >= stn["lap_start"])
                 & (laps["lap_number"] <= stn["lap_end"]))
            laps.loc[m, "Stint"] = stn["stint_number"]
            laps.loc[m, "Compound"] = stn.get("compound")
            age0 = stn.get("tyre_age_at_start") or 0
            laps.loc[m, "TyreLife"] = age0 + (laps.loc[m, "lap_number"] - stn["lap_start"]) + 1
            # last lap of a stint is the in-lap to the pits
            laps.loc[(laps["driver_number"] == stn["driver_number"])
                     & (laps["lap_number"] == stn["lap_end"]), "PitInTime"] = 1

    out = pd.DataFrame({
        "Driver": laps["name_acronym"].fillna(laps["driver_number"].astype(str)),
        "Team": laps["team_name"].fillna("Unknown"),
        "LapNumber": laps["lap_number"],
        "Stint": laps["Stint"],
        "Compound": laps["Compound"],
        "TyreLife": laps["TyreLife"],
        "PitInTime": laps["PitInTime"],
        "PitOutTime": np.where(laps.get("is_pit_out_lap", False) == True, 1, np.nan),
        "LapSeconds": pd.to_numeric(laps["lap_duration"], errors="coerce"),
    })
    out["LapTime"] = pd.to_timedelta(out["LapSeconds"], unit="s")
    out["IsAccurate"] = out["LapSeconds"].notna() & out["PitOutTime"].isna()
    out = out.dropna(subset=["LapSeconds"])
    if out.empty:
        raise ValueError("OpenF1 laps have no lap times yet")
    return out[LAP_COLUMNS]


@st.cache_data(ttl=3600, show_spinner=False)
def _actual_results_fastf1(year, round_number):
    sess = fastf1.get_session(year, round_number, "Race")
    sess.load(laps=False, telemetry=False, weather=False, messages=False)
    res = sess.results[["Abbreviation", "Position"]].copy()
    res["Position"] = pd.to_numeric(res["Position"], errors="coerce")
    res = res.dropna(subset=["Position"])
    if res.empty:
        raise ValueError("No race results yet")
    return res.rename(columns={"Abbreviation": "Driver", "Position": "ActualFinish"})


@st.cache_data(ttl=3600, show_spinner=False)
def _actual_results_openf1(year, race_date_utc):
    sessions = _openf1_get("sessions", year=year, session_name="Race")
    target = pd.Timestamp(race_date_utc)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    best = min(sessions, key=lambda x: abs(pd.Timestamp(x["date_start"]).tz_convert("UTC") - target))
    if abs(pd.Timestamp(best["date_start"]).tz_convert("UTC") - target) > pd.Timedelta(days=2):
        raise ValueError("Race not found on OpenF1")
    key = best["session_key"]
    res = pd.DataFrame(_openf1_get("session_result", session_key=key))
    time.sleep(0.4)
    drivers = pd.DataFrame(_openf1_get("drivers", session_key=key))
    if res.empty:
        raise ValueError("No race results yet")
    drivers = drivers.drop_duplicates("driver_number")[["driver_number", "name_acronym"]]
    res = res.merge(drivers, on="driver_number", how="left")
    res["ActualFinish"] = pd.to_numeric(res["position"], errors="coerce")
    res = res.dropna(subset=["ActualFinish"])
    return res.rename(columns={"name_acronym": "Driver"})[["Driver", "ActualFinish"]]


def load_actual_results(year, round_number, event):
    """Real race result for a completed race, or None. Tries FastF1 then OpenF1."""
    race_date = session_date_for(event, "Race")
    if race_date is None:
        return None
    rd = race_date.tz_localize("UTC") if race_date.tzinfo is None else race_date.tz_convert("UTC")
    if rd + pd.Timedelta(hours=3) > pd.Timestamp.now(tz="UTC"):
        return None  # race not finished yet
    try:
        return _actual_results_fastf1(year, round_number)
    except Exception:
        pass
    try:
        return _actual_results_openf1(year, race_date.isoformat())
    except Exception:
        return None


def session_date_for(event, session_name):
    for i in range(1, 6):
        if event.get(f"Session{i}") == session_name:
            for col in (f"Session{i}DateUtc", f"Session{i}Date"):
                if col in event.index and pd.notna(event[col]):
                    return pd.Timestamp(event[col])
    return None


def load_session_laps(year, round_number, session_name, event):
    """Returns (laps, source, error_message). Tries FastF1, then OpenF1."""
    errors = []
    try:
        return _load_fastf1_laps(year, round_number, session_name), "FastF1", None
    except Exception as e:
        errors.append(f"FastF1 - {type(e).__name__}: {e}")
    sdate = session_date_for(event, session_name)
    if sdate is None:
        errors.append("OpenF1 - this weekend has no such session (sprint weekend?)")
        return None, None, " || ".join(errors)
    try:
        return _load_openf1_laps(year, session_name, sdate.isoformat()), "OpenF1", None
    except Exception as e:
        errors.append(f"OpenF1 - {type(e).__name__}: {e}")
    return None, None, " || ".join(errors)


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

st.sidebar.title("⚙️ Settings")
this_year = datetime.now().year
year = int(st.sidebar.selectbox("Season", list(range(this_year, FIRST_SEASON - 1, -1))))
events, default_idx = selectable_events(year)
labels = [f"R{int(r.RoundNumber)} – {r.EventName}" for r in events.itertuples()]
pick = st.sidebar.selectbox("Track", range(len(labels)), index=default_idx,
                            format_func=lambda i: labels[i])
event = events.iloc[pick]
FP_CHOICES = {
    "All sessions (FP1 + FP2 + FP3)": ["FP1", "FP2", "FP3"],
    "FP1 only": ["FP1"],
    "FP2 only": ["FP2"],
    "FP3 only": ["FP3"],
    "FP1 + FP2": ["FP1", "FP2"],
    "FP2 + FP3": ["FP2", "FP3"],
}
fp_pick = st.sidebar.radio("Practice data to use", list(FP_CHOICES.keys()), index=0,
                           help="Only the selected sessions feed the qualifying and race projections.")
selected_fp = FP_CHOICES[fp_pick]

grid_weight = st.sidebar.slider(
    "Grid position weight", 0, 60, GRID_WEIGHT_DEFAULT, step=5, format="%d%%",
    help="How much the projected qualifying position counts in the race prediction. "
         "0% means race prediction uses practice long-run pace only.") / 100
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

round_number = int(event["RoundNumber"])
circuit = event["Location"]
st.subheader(f"{event['EventName']} – {year}, Round {round_number}")
st.caption(f"{circuit}, {event['Country']}")

ALL_FP = {"FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"}
sess_map = {k: v for k, v in ALL_FP.items() if k in selected_fp}
session_laps = {}
load_errors = {}
sources = {}
with st.spinner("Loading practice data..."):
    for short, full in sess_map.items():
        laps, src, err = load_session_laps(year, round_number, full, event)
        session_laps[short] = laps
        if err:
            load_errors[short] = err
        else:
            sources[short] = src

available = [k for k, v in session_laps.items() if v is not None and not v.empty]

with st.sidebar.expander("Diagnostics", expanded=not available):
    st.write(f"FastF1 version: {fastf1.__version__}")
    st.write(f"Event picked: {event['EventName']} (round {round_number})")
    for short in sess_map:
        if short in load_errors:
            st.error(f"{short}: {load_errors[short]}")
        else:
            st.success(f"{short}: {len(session_laps[short])} laps from {sources[short]}")

if not available:
    st.warning("No practice data loaded for this race. Open Diagnostics in the sidebar "
               "to see why, then tap Refresh data to try again. Note OpenF1 only covers 2023 onward.")
    for short, err in load_errors.items():
        st.code(f"{short}: {err}")
    st.stop()
st.info("Practice sessions used: " + ", ".join(available))

quali = quali_projection(session_laps)
pace = long_run_pace(session_laps)
race = race_projection(quali, pace, grid_weight) if quali is not None else None

actual = load_actual_results(year, round_number, event)
if race is not None and actual is not None:
    race = race.merge(actual, on="Driver", how="left")

tab1, tab2, tab3 = st.tabs(["🏆 Podium", "⏱️ Qualifying", "🏎️ Race"])

with tab1:
    if race is None or race.empty:
        st.write("Not enough long-run data yet to project the race.")
    else:
        top3 = race.head(3)
        cols = st.columns(3)
        medals = ["🥇 P1", "🥈 P2", "🥉 P3"]
        for c, m, (_, r) in zip(cols, medals, top3.iterrows()):
            c.metric(m, r["Driver"], r["Team"], delta_color="off")
        st.caption(f"Predicted podium. Grid position weight: {grid_weight:.0%}")
        if actual is not None:
            st.markdown("**Actual podium**")
            real = actual.sort_values("ActualFinish").head(3)
            acols = st.columns(3)
            pred3 = set(top3["Driver"])
            for c, m, (_, r) in zip(acols, medals, real.iterrows()):
                c.metric(m, r["Driver"], "✓ predicted" if r["Driver"] in pred3 else "missed",
                         delta_color="normal" if r["Driver"] in pred3 else "inverse")
            hits = len(pred3 & set(real["Driver"]))
            exact = sum(a == b for a, b in zip(top3["Driver"], real["Driver"]))
            st.success(f"{hits} of 3 podium drivers predicted, {exact} in the exact position.")

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
        cols_shown = ["ProjectedFinish", "Driver", "Team", "ProjectedGrid"]
        if "ActualFinish" in race.columns:
            cols_shown.append("ActualFinish")
        cols_shown += ["RaceGapPct", "DegPerLap", "LongRunLaps"]
        show = race[cols_shown].copy()
        if "ActualFinish" in show.columns:
            show["ActualFinish"] = show["ActualFinish"].astype("Int64")
            show = show.rename(columns={"ActualFinish": "Actual finish"})
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

st.caption("Built from FastF1 and OpenF1 data. Projections are estimates, not guarantees.")
