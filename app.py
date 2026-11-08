import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import base64
import re
import html as htmllib
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import time
from datetime import datetime

st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏆", layout="wide")
fastf1.Cache.enable_cache("/tmp")

QUALI_WEIGHTS = {"FP1": 0.20, "FP2": 0.35, "FP3": 0.45}
FIRST_SEASON = 2021
GRID_WEIGHT_DEFAULT = 25  # percent; adjustable in the sidebar

# Typical lap-time difference vs the medium tyre, in seconds per lap.
# Used only when a weekend doesn't have enough mixed-compound runs to measure it.
COMPOUND_DEFAULTS = {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.4}
COMPOUND_LIMITS = {"SOFT": (-1.2, -0.1), "HARD": (0.1, 1.2)}
DRY_COMPOUNDS = set(COMPOUND_DEFAULTS)


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
               "TyreLife", "IsAccurate", "PitInTime", "PitOutTime", "LapSeconds", "TeamColor"]


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
    laps["TeamColor"] = np.nan
    try:  # official live-timing team colour
        colours = session.results[["Abbreviation", "TeamColor"]].dropna()
        laps["TeamColor"] = laps["Driver"].map(dict(zip(colours["Abbreviation"], colours["TeamColor"])))
    except Exception:
        pass
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

    if "team_colour" not in drivers.columns:
        drivers["team_colour"] = np.nan
    drivers = drivers.drop_duplicates("driver_number")[["driver_number", "name_acronym", "team_name", "team_colour"]]
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
        "TeamColor": laps["team_colour"],
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
def _result_fastf1(year, round_number, session_name):
    sess = fastf1.get_session(year, round_number, session_name)
    sess.load(laps=False, telemetry=False, weather=False, messages=False)
    res = sess.results[["Abbreviation", "Position"]].copy()
    res["Position"] = pd.to_numeric(res["Position"], errors="coerce")
    res = res.dropna(subset=["Position"])
    if res.empty:
        raise ValueError(f"No {session_name} results yet")
    return res.rename(columns={"Abbreviation": "Driver"})[["Driver", "Position"]]


@st.cache_data(ttl=3600, show_spinner=False)
def _result_openf1(year, session_name, session_date_utc):
    sessions = _openf1_get("sessions", year=year, session_name=session_name)
    if not sessions:
        raise ValueError(f"No {session_name} sessions on OpenF1")
    target = pd.Timestamp(session_date_utc)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    best = min(sessions, key=lambda x: abs(pd.Timestamp(x["date_start"]).tz_convert("UTC") - target))
    if abs(pd.Timestamp(best["date_start"]).tz_convert("UTC") - target) > pd.Timedelta(days=2):
        raise ValueError(f"{session_name} not found on OpenF1")
    key = best["session_key"]
    res = pd.DataFrame(_openf1_get("session_result", session_key=key))
    time.sleep(0.4)
    drivers = pd.DataFrame(_openf1_get("drivers", session_key=key))
    if res.empty:
        raise ValueError(f"No {session_name} results yet")
    drivers = drivers.drop_duplicates("driver_number")[["driver_number", "name_acronym"]]
    res = res.merge(drivers, on="driver_number", how="left")
    res["Position"] = pd.to_numeric(res["position"], errors="coerce")
    res = res.dropna(subset=["Position"])
    return res.rename(columns={"name_acronym": "Driver"})[["Driver", "Position"]]


def load_session_result(year, round_number, event, session_name, done_after_hours):
    """Real classification for a finished session, or None. Tries FastF1 then OpenF1."""
    sdate = session_date_for(event, session_name)
    if sdate is None:
        return None
    sd = sdate.tz_localize("UTC") if sdate.tzinfo is None else sdate.tz_convert("UTC")
    if sd + pd.Timedelta(hours=done_after_hours) > pd.Timestamp.now(tz="UTC"):
        return None  # session not finished yet
    for loader in (lambda: _result_fastf1(year, round_number, session_name),
                   lambda: _result_openf1(year, session_name, sdate.isoformat())):
        try:
            res = loader()
            return res.sort_values("Position").reset_index(drop=True)
        except Exception:
            continue
    return None


@st.cache_data(ttl=600, show_spinner=False)
def _grid_fastf1(year, round_number):
    """Official grid as the race actually started (only exists after the race)."""
    sess = fastf1.get_session(year, round_number, "Race")
    sess.load(laps=False, telemetry=False, weather=False, messages=False)
    res = sess.results[["Abbreviation", "GridPosition"]].copy()
    res["GridPosition"] = pd.to_numeric(res["GridPosition"], errors="coerce")
    res = res.dropna(subset=["GridPosition"])
    if res.empty or (res["GridPosition"] == 0).all():
        raise ValueError("No grid in race results yet")
    # 0 means a pit-lane start: put those drivers at the back
    back = int(res["GridPosition"].max()) + 1
    res.loc[res["GridPosition"] == 0, "GridPosition"] = back
    res = res.sort_values("GridPosition").reset_index(drop=True)
    res["Position"] = range(1, len(res) + 1)
    return res.rename(columns={"Abbreviation": "Driver"})[["Driver", "Position"]]


@st.cache_data(ttl=600, show_spinner=False)
def _grid_openf1(year, race_date_utc):
    """FIA starting grid with penalties applied, published before the race."""
    sessions = _openf1_get("sessions", year=year, session_name="Race")
    if not sessions:
        raise ValueError("No races on OpenF1")
    target = pd.Timestamp(race_date_utc)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    race = min(sessions, key=lambda x: abs(pd.Timestamp(x["date_start"]).tz_convert("UTC") - target))
    if abs(pd.Timestamp(race["date_start"]).tz_convert("UTC") - target) > pd.Timedelta(days=2):
        raise ValueError("Race not found on OpenF1")
    grid = []
    for params in ({"meeting_key": race["meeting_key"]}, {"session_key": race["session_key"]}):
        try:
            grid = _openf1_get("starting_grid", **params)
        except Exception:
            grid = []
        if grid:
            break
        time.sleep(0.4)
    if not grid:
        raise ValueError("Starting grid not published yet")
    g = pd.DataFrame(grid)
    if "session_key" in g.columns and g["session_key"].nunique() > 1:
        g = g[g["session_key"] == g["session_key"].max()]
    time.sleep(0.4)
    drivers = pd.DataFrame(_openf1_get("drivers", session_key=g["session_key"].iloc[0]))
    drivers = drivers.drop_duplicates("driver_number")[["driver_number", "name_acronym"]]
    g = g.merge(drivers, on="driver_number", how="left")
    g["Position"] = pd.to_numeric(g["position"], errors="coerce")
    g = g.dropna(subset=["Position", "name_acronym"]).sort_values("Position")
    return g.rename(columns={"name_acronym": "Driver"})[["Driver", "Position"]].reset_index(drop=True)


def load_starting_grid(year, round_number, event):
    """Official starting grid including FIA grid penalties, or None if not out yet."""
    race_date = session_date_for(event, "Race")
    if race_date is None:
        return None
    rd = race_date.tz_localize("UTC") if race_date.tzinfo is None else race_date.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    loaders = []
    if rd + pd.Timedelta(hours=3) < now:
        loaders.append(lambda: _grid_fastf1(year, round_number))
    loaders.append(lambda: _grid_openf1(year, race_date.isoformat()))
    for loader in loaders:
        try:
            return loader()
        except Exception:
            continue
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


def _stint_runs(session_laps, min_laps=5):
    """Every clean run of at least min_laps laps, with pace corrected to 5-lap-old tyres.
    Returns (one row per run, one row per lap of those runs)."""
    rows = []
    lap_rows = []
    for name, laps in session_laps.items():
        if laps is None or laps.empty:
            continue
        clean = laps[(laps["IsAccurate"] == True) & laps["LapSeconds"].notna()
                     & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
                     & laps["TyreLife"].notna()
                     & laps["Compound"].astype(str).str.upper().isin(DRY_COMPOUNDS)]
        for (drv, team, stint), grp in clean.groupby(["Driver", "Team", "Stint"]):
            if len(grp) < min_laps:
                continue
            med = grp["LapSeconds"].median()
            grp = grp[grp["LapSeconds"] < med * 1.03]
            if len(grp) < min_laps:
                continue
            slope = np.polyfit(grp["TyreLife"].astype(float), grp["LapSeconds"], 1)[0]
            slope = max(min(slope, 0.2), -0.05)
            avg_age = grp["TyreLife"].mean()
            corrected = grp["LapSeconds"].mean() - slope * (avg_age - 5)
            compound = str(grp["Compound"].mode().iloc[0]).upper()
            rows.append({"Driver": drv, "Team": team, "Session": name, "Laps": len(grp),
                         "Compound": compound, "Pace": corrected, "DegPerLap": slope})
            g = grp.sort_values("LapNumber")[["Driver", "Team", "LapNumber", "TyreLife", "LapSeconds"]].copy()
            g["Session"] = name
            g["Stint"] = stint
            g["Compound"] = compound
            g["RunLap"] = range(1, len(g) + 1)
            g["Run"] = f"{drv} {name} {compound.title()}"
            lap_rows.append(g)
    runlaps = pd.concat(lap_rows, ignore_index=True) if lap_rows else pd.DataFrame()
    return pd.DataFrame(rows), runlaps


def estimate_compound_offsets(runs):
    """Learns soft/hard deltas vs medium, and how much faster the track got from
    session to session, from this weekend's long runs. Fits
    pace = driver + session + compound with least squares, so a driver's own runs
    on different tyres (or in different sessions) are what reveal the gaps.
    Compounds without at least 2 drivers running them alongside another compound
    use the typical defaults instead."""
    offsets = dict(COMPOUND_DEFAULTS)
    source = {c: "typical" for c in COMPOUND_DEFAULTS}
    source["MEDIUM"] = "reference"
    session_shift = {}
    if runs is None or runs.empty:
        return offsets, source, session_shift

    per_driver = runs.groupby("Driver")["Compound"].nunique()
    mixed = set(per_driver[per_driver > 1].index)
    measurable = [c for c in ("SOFT", "HARD")
                  if runs[(runs["Compound"] == c) & runs["Driver"].isin(mixed)]["Driver"].nunique() >= 2]

    df = runs.copy()
    # pre-correct compounds we can't measure, then fit the rest
    fixed = df["Compound"].map(lambda c: 0.0 if c in measurable else COMPOUND_DEFAULTS.get(c, 0.0))
    y = (df["Pace"] - fixed).to_numpy()
    drivers = sorted(df["Driver"].unique())
    sessions = sorted(df["Session"].unique())
    X = []
    for _, r in df.iterrows():
        row = [1.0 if r["Driver"] == d else 0.0 for d in drivers]
        row += [1.0 if r["Session"] == s_ else 0.0 for s_ in sessions[1:]]
        row += [1.0 if r["Compound"] == c else 0.0 for c in measurable]
        X.append(row)
    X = np.array(X)
    if X.shape[1] == len(drivers):  # one session and nothing to measure
        return offsets, source, session_shift
    w = np.sqrt(df["Laps"].to_numpy(dtype=float))  # longer runs count more
    coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)

    k = len(drivers)
    # only trust a session shift if enough drivers did long runs in both sessions
    first_drivers = set(df[df["Session"] == sessions[0]]["Driver"])
    for i, s_ in enumerate(sessions[1:]):
        shared = first_drivers & set(df[df["Session"] == s_]["Driver"])
        if len(shared) >= 3:
            session_shift[s_] = float(min(max(coef[k + i], -2.0), 2.0))
    k += len(sessions) - 1
    for i, c in enumerate(measurable):
        lo, hi = COMPOUND_LIMITS[c]
        offsets[c] = float(min(max(coef[k + i], lo), hi))
        source[c] = "measured"
    return offsets, source, session_shift


def _medium_equivalent(laps, offsets):
    comp = laps["Compound"].astype(str).str.upper()
    return laps["LapSeconds"] - comp.map(offsets).fillna(0.0)


def quali_projection(session_laps, offsets):
    rows = []
    for name, laps in session_laps.items():
        if laps is None or laps.empty:
            continue
        dry = laps[laps["Compound"].astype(str).str.upper().isin(DRY_COMPOUNDS)
                   & laps["LapSeconds"].notna()].copy()
        if dry.empty:
            continue
        dry["AdjSeconds"] = _medium_equivalent(dry, offsets)
        best = dry.loc[dry.groupby("Driver")["AdjSeconds"].idxmin(),
                       ["Driver", "Team", "AdjSeconds", "Compound"]]
        fastest = best["AdjSeconds"].min()
        best["GapPct"] = (best["AdjSeconds"] - fastest) / fastest * 100
        best["Session"] = name
        rows.append(best)
    if not rows:
        return None
    allbest = pd.concat(rows)
    pivot = allbest.pivot_table(index=["Driver", "Team"], columns="Session",
                                values="GapPct").reset_index()
    score = pd.Series(0.0, index=pivot.index)
    total_w = pd.Series(0.0, index=pivot.index)
    for s_, w in QUALI_WEIGHTS.items():
        if s_ in pivot.columns:
            has = pivot[s_].notna()
            score[has] += pivot.loc[has, s_] * w
            total_w[has] += w
    pivot["QualiScore"] = score / total_w.replace(0, np.nan)
    pivot = pivot.dropna(subset=["QualiScore"]).sort_values("QualiScore").reset_index(drop=True)
    pivot["ProjectedGrid"] = range(1, len(pivot) + 1)
    return pivot


def long_run_pace(runs, offsets, session_shift=None):
    if runs is None or runs.empty:
        return None
    runs = runs.copy()
    runs["Pace"] = runs["Pace"] - runs["Compound"].map(offsets).fillna(0.0)
    if session_shift:
        runs["Pace"] = runs["Pace"] - runs["Session"].map(session_shift).fillna(0.0)
    best_runs = runs.sort_values("Laps", ascending=False).groupby("Driver").head(2)
    agg = best_runs.groupby(["Driver", "Team"]).agg(
        RacePace=("Pace", "mean"), DegPerLap=("DegPerLap", "mean"),
        LongRunLaps=("Laps", "sum"),
        Tyres=("Compound", lambda x: "/".join(sorted(set(x))))).reset_index()
    fastest = agg["RacePace"].min()
    agg["RaceGapPct"] = (agg["RacePace"] - fastest) / fastest * 100
    return agg


def race_projection(quali, pace, grid_weight, known_grid=None):
    """Blends long-run pace with grid position. Uses the best grid available:
    official starting grid (with penalties) > real qualifying > projected."""
    if pace is None:
        return None
    if known_grid is not None:
        grid = known_grid.rename(columns={"Position": "ProjectedGrid"})[["Driver", "ProjectedGrid"]]
    elif quali is not None:
        grid = quali[["Driver", "ProjectedGrid"]]
    else:
        grid = pd.DataFrame({"Driver": pace["Driver"], "ProjectedGrid": np.nan})
    df = pace.merge(grid, on="Driver", how="left")
    if df["ProjectedGrid"].isna().all():
        df["ProjectedGrid"] = len(df)
    df["ProjectedGrid"] = df["ProjectedGrid"].fillna(df["ProjectedGrid"].max() + 1)
    pace_rank = df["RaceGapPct"].rank()
    df["RaceScore"] = (1 - grid_weight) * pace_rank + grid_weight * df["ProjectedGrid"]
    df = df.sort_values("RaceScore").reset_index(drop=True)
    df["ProjectedFinish"] = df.index + 1
    return df


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def track_outline_svg(year, session_name, session_date_utc):
    """Traces one fast lap from OpenF1 car-position data and returns the circuit
    shape as an SVG string (or None). OpenF1 position data starts in 2023."""
    sessions = _openf1_get("sessions", year=year, session_name=session_name)
    if not sessions:
        return None
    target = pd.Timestamp(session_date_utc)
    target = target.tz_localize("UTC") if target.tzinfo is None else target.tz_convert("UTC")
    best = min(sessions, key=lambda x: abs(pd.Timestamp(x["date_start"]).tz_convert("UTC") - target))
    if abs(pd.Timestamp(best["date_start"]).tz_convert("UTC") - target) > pd.Timedelta(days=2):
        return None
    key = best["session_key"]
    laps = pd.DataFrame(_openf1_get("laps", session_key=key))
    if laps.empty or "lap_duration" not in laps.columns:
        return None
    laps = laps.dropna(subset=["lap_duration", "date_start"])
    if "is_pit_out_lap" in laps.columns:
        laps = laps[laps["is_pit_out_lap"] != True]
    if laps.empty:
        return None
    lap = laps.sort_values("lap_duration").iloc[0]
    start = pd.Timestamp(lap["date_start"]).tz_convert("UTC")
    end = start + pd.Timedelta(seconds=float(lap["lap_duration"]))
    fmt = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S")
    # OpenF1 wants the > and < operators unencoded in the URL
    url = (f"{OPENF1}/location?session_key={key}&driver_number={int(lap['driver_number'])}"
           f"&date>{fmt(start)}&date<{fmt(end)}")
    time.sleep(0.4)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    pts = pd.DataFrame(r.json())
    if pts.empty or not {"x", "y"} <= set(pts.columns):
        return None
    pts = pts[(pts["x"] != 0) | (pts["y"] != 0)]
    if len(pts) < 50:
        return None
    x, y = pts["x"].to_numpy(float), pts["y"].to_numpy(float)
    if (y.max() - y.min()) > (x.max() - x.min()):  # lay tall tracks on their side
        x, y = y, -x
    w, h = x.max() - x.min(), y.max() - y.min()
    scale = 1000 / max(w, h)
    pad = 90  # breathing room so the outline never touches the panel edges
    px_ = (x - x.min()) * scale + pad
    py_ = (y.max() - y) * scale + pad  # SVG y runs downward
    vw, vh = w * scale + 2 * pad, h * scale + 2 * pad
    path = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px_, py_))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:.0f} {vh:.0f}">'
            f'<polyline points="{path}" fill="none" stroke="#3E6F40" stroke-opacity="0.06" '
            f'stroke-width="26" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<polyline points="{path}" fill="none" stroke="#3E6F40" stroke-opacity="0.10" '
            f'stroke-width="7" stroke-linejoin="round" stroke-linecap="round"/></svg>')


def set_track_background(event, year):
    """Puts the selected circuit's outline faintly behind the page."""
    svg = None
    for sname in ("Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"):
        sdate = session_date_for(event, sname)
        if sdate is None:
            continue
        if (sdate.tz_localize("UTC") if sdate.tzinfo is None else sdate) > pd.Timestamp.now(tz="UTC"):
            continue
        try:
            svg = track_outline_svg(year, sname, sdate.isoformat())
        except Exception:
            svg = None
        if svg:
            break
    if not svg:
        return
    b64 = base64.b64encode(svg.encode()).decode()
    st.markdown(f"""<style>
    /* main panel only, so the track is centred in the space beside Settings.
       The main panel is the scrolling element, so the track stays put while you scroll */
    [data-testid="stMain"], section.main {{
        background-image: url("data:image/svg+xml;base64,{b64}");
        background-repeat: no-repeat;
        background-position: center center;
        background-size: contain;
        background-attachment: scroll;
    }}
    </style>""", unsafe_allow_html=True)


def team_colour_map(session_laps):
    """Team -> '#rrggbb' using the colours F1 live timing publishes."""
    frames = [v[["Team", "TeamColor"]] for v in session_laps.values()
              if v is not None and not v.empty and "TeamColor" in v.columns]
    if not frames:
        return {}
    df = pd.concat(frames).dropna()
    out = {}
    for team, col in df.groupby("Team")["TeamColor"].first().items():
        col = str(col).strip().lstrip("#")
        if re.fullmatch(r"[0-9a-fA-F]{6}", col):
            out[team] = "#" + col
    return out


def teammate_dash(drivers_teams):
    """Second driver in each team gets a dashed line, like the timing screens."""
    dash, seen = {}, {}
    for drv, team in sorted(drivers_teams):
        seen[team] = seen.get(team, 0) + 1
        dash[drv] = "solid" if seen[team] == 1 else "dash"
    return dash


# ---------- Pirelli strategy + announced penalties (from strategies.json) ----------
def load_race_notes(year, round_number):
    """Reads this race's entry from strategies.json in the repo, if there is one."""
    path = Path(__file__).parent / "strategies.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get(f"{year}-{round_number}", {})


PIRELLI_FEED = "https://press.pirelli.com/tagfeed/en/tags/{year}__season"
_STRAT_WORDS = re.compile(r"\b(one|two|three|single|1|2|3)[- ]?stop|\bstop strategy|\bstops?\b", re.I)
_COMPOUND_WORDS = re.compile(r"\b(soft|medium|hard)s?\b", re.I)
_PICK_WORDS = re.compile(r"preferred|quickest|fastest|optimal|optimum|best|favou?r|recommended|most effective", re.I)


def _plain(text):
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _html_to_text(raw):
    txt = re.sub(r"<[^>]+>", " ", raw or "")
    return re.sub(r"\s+", " ", htmllib.unescape(txt)).strip()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_pirelli_items(year):
    r = requests.get(PIRELLI_FEED.format(year=year), timeout=20,
                     headers={"User-Agent": "Mozilla/5.0 (F1 Podium Predictor)"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        get = lambda tag: (it.findtext(tag) or "")
        body = get("description")
        content = it.find("{http://purl.org/rss/1.0/modules/content/}encoded")
        if content is not None and content.text:
            body = content.text
        try:
            published = pd.Timestamp(pd.to_datetime(get("pubDate"), utc=True))
        except Exception:
            published = None
        items.append({"title": get("title"), "link": get("link"), "published": published,
                      "categories": get("category"), "text": _html_to_text(body)})
    return items


def pirelli_strategy(year, event):
    """Finds Pirelli's latest pre-race release for this weekend and pulls out the
    strategy sentences. Returns a dict like strategies.json entries, or None."""
    try:
        items = fetch_pirelli_items(year)
    except Exception:
        return None
    race_date = session_date_for(event, "Race")
    first = session_date_for(event, "Practice 1") or race_date
    if race_date is None:
        return None
    to_utc = lambda t: t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    race_date, first = to_utc(race_date), to_utc(first)
    name = _plain(str(event["EventName"]).replace("Grand Prix", ""))
    loc = _plain(str(event["Location"]))

    candidates = []
    for it in items:
        pub = it["published"]
        if pub is None or not (first - pd.Timedelta(days=6) <= pub <= race_date):
            continue  # only previews and Friday/Saturday notes, not the race report
        tag_text = _plain(it["categories"] + " " + it["title"] + " " + it["text"][:600])
        named = name.strip() in tag_text or loc in tag_text
        sentences = re.split(r"(?<=[.!?])\s+", it["text"])
        strat = [x for x in sentences if _STRAT_WORDS.search(x) and _COMPOUND_WORDS.search(x)
                 and len(x) < 400]
        if strat and named:
            candidates.append((pub, it, strat))
    if not candidates:
        return None
    pub, it, strat = max(candidates, key=lambda c: c[0])  # latest = most informed
    headline = next((x for x in strat if _PICK_WORDS.search(x)), strat[0])
    compounds = sorted({m.upper() for m in _COMPOUND_WORDS.findall(headline)},
                       key=["SOFT", "MEDIUM", "HARD"].index)
    details = [x for x in strat if x != headline][:3]
    return {"strategy": headline, "details": details, "compounds": compounds,
            "source": f"Pirelli press release, {pub.strftime('%a %d %b')} (auto-imported)",
            "url": it["link"], "title": it["title"]}


def apply_grid_drops(grid, drops):
    """Applies announced penalties to a qualifying order.
    drops: {"HAM": 10} = 10 places back, {"HAM": "back"} = back of the grid."""
    if grid is None or not drops:
        return grid
    g = grid.copy()
    n = len(g)
    keys = []
    for i, r in enumerate(g.itertuples()):
        d = drops.get(r.Driver)
        if d is None:
            keys.append(r.Position)
        elif isinstance(d, str):          # "back" or "pitlane"
            keys.append(n + 100 + (1 if d.lower() == "pitlane" else 0) + i / 1000)
        else:                             # N places, sits behind the car it lands on
            keys.append(r.Position + int(d) + 0.5)
    g["_k"] = keys
    g = g.sort_values("_k").drop(columns="_k").reset_index(drop=True)
    g["Position"] = range(1, len(g) + 1)
    return g


# ---------- Colour palette ----------
PALETTE = {"forest": "#3E6F40", "mint": "#CFFFDC", "leaf": "#68BA7F", "pine": "#253D2C"}


def apply_palette():
    p = PALETTE
    st.markdown(f"""<style>
    h1, h2, h3, h4 {{ color: {p['pine']} !important; }}
    h1 {{ border-bottom: 3px solid {p['leaf']}; padding-bottom: .3rem; }}
    [data-testid="stSidebar"] {{ border-right: 2px solid {p['leaf']}; }}
    [data-testid="stSidebar"] h1 {{ border-bottom: none; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {p['leaf']} !important;
        background: rgba(207, 255, 220, 0.35);
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background: #ffffff;
    }}
    [data-testid="stMetric"] {{
        background: {p['mint']}; border: 1px solid {p['leaf']};
        border-radius: 10px; padding: .6rem .8rem;
    }}
    [data-testid="stMetricValue"] {{ color: {p['pine']}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
    .stTabs [data-baseweb="tab"] {{
        background: {p['mint']}; border-radius: 8px 8px 0 0; padding: .4rem .9rem;
    }}
    .stTabs [data-baseweb="tab"] p {{ color: {p['pine']}; }}
    .stTabs [aria-selected="true"] {{ background: {p['forest']}; }}
    .stTabs [aria-selected="true"] p {{ color: #ffffff !important; font-weight: 700; }}
    .stButton > button {{
        background: {p['forest']}; color: #ffffff; border: none; font-weight: 700;
    }}
    .stButton > button:hover {{ background: {p['leaf']}; color: #ffffff; }}
    [data-testid="stExpander"] details {{ border-color: {p['leaf']} !important; }}
    </style>""", unsafe_allow_html=True)


def style_fig(fig):
    """White chart backgrounds with palette-coloured text and grid lines."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.85)",
                      font_color=PALETTE["pine"], title_font_color=PALETTE["pine"])
    fig.update_xaxes(gridcolor="rgba(62,111,64,0.15)", zerolinecolor="rgba(62,111,64,0.35)",
                     linecolor="rgba(62,111,64,0.4)")
    fig.update_yaxes(gridcolor="rgba(62,111,64,0.15)", zerolinecolor="rgba(62,111,64,0.35)",
                     linecolor="rgba(62,111,64,0.4)")
    return fig


# ---------- UI ----------
apply_palette()
st.title("F1 Podium Predictor")

st.sidebar.title("⚙️ Settings")
this_year = datetime.now().year
year = int(st.sidebar.selectbox("Season", list(range(this_year, FIRST_SEASON - 1, -1))))
events, default_idx = selectable_events(year)
labels = [f"R{int(r.RoundNumber)} – {r.EventName}" for r in events.itertuples()]
pick = st.sidebar.selectbox("Track", range(len(labels)), index=default_idx,
                            format_func=lambda i: labels[i])
event = events.iloc[pick]
race_notes = load_race_notes(year, int(event["RoundNumber"]))
if not race_notes.get("strategy"):
    auto = pirelli_strategy(year, event)
    if auto:
        race_notes = {**race_notes, **auto}

data_box = st.sidebar.container(border=True)  # "Data used" box: tyre dropdown + summary
data_lines = []
pirelli_plan = {c.upper() for c in race_notes.get("compounds", [])} & {"SOFT", "MEDIUM", "HARD"}
TYRE_CHOICES = {"All compounds": {"SOFT", "MEDIUM", "HARD"}}
if pirelli_plan:
    label = " + ".join(c.title() for c in ("SOFT", "MEDIUM", "HARD") if c in pirelli_plan)
    TYRE_CHOICES[f"Match Pirelli strategy ({label})"] = pirelli_plan
TYRE_CHOICES.update({
    "Soft only": {"SOFT"}, "Medium only": {"MEDIUM"}, "Hard only": {"HARD"},
    "Soft + Medium": {"SOFT", "MEDIUM"}, "Medium + Hard": {"MEDIUM", "HARD"},
    "Soft + Hard": {"SOFT", "HARD"},
})
with data_box:
    st.markdown("**Data used**")
    tyre_pick = st.selectbox("Tyre type", list(TYRE_CHOICES.keys()), index=0,
                             help="Which tyre compounds' laps feed the projections.")
selected_compounds = set(TYRE_CHOICES[tyre_pick])
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


min_long_run = st.sidebar.slider(
    "Minimum long-run length (laps)", 3, 15, 5,
    help="A run on one set of tyres needs at least this many clean laps to count as a long run. "
         "Higher = closer to race conditions but fewer drivers qualify.")

grid_weight = st.sidebar.slider(
    "Grid position weight", 0, 60, GRID_WEIGHT_DEFAULT, step=5, format="%d%%",
    help="How much the projected qualifying position counts in the race prediction. "
         "0% means race prediction uses practice long-run pace only.") / 100
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

round_number = int(event["RoundNumber"])
set_track_background(event, year)
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
data_lines.append("Sessions: " + ", ".join(available))

if not selected_compounds:
    st.warning("Pick a tyre type in Settings.")
    st.stop()
session_laps_all = dict(session_laps)  # every compound, for the Long runs tab
# keep only laps on the chosen compounds
session_laps = {
    k: (v[v["Compound"].astype(str).str.upper().isin(selected_compounds)] if v is not None else None)
    for k, v in session_laps.items()
}
if all(v is None or v.empty for v in session_laps.values()):
    st.warning("No laps on the selected tyre type in these sessions. Try another option under Tyre type.")
    st.stop()
data_lines.append("Tyres: " + ", ".join(c.title() for c in ("SOFT", "MEDIUM", "HARD")
                                          if c in selected_compounds))

runs, runlaps = _stint_runs(session_laps, min_long_run)
runs_all, runlaps_all = _stint_runs(session_laps_all, min_long_run)
offsets_all, _, shift_all = estimate_compound_offsets(runs_all)
TEAM_COLOURS = team_colour_map(session_laps_all)
compound_offsets, compound_source, session_shift = estimate_compound_offsets(runs)
quali = quali_projection(session_laps, compound_offsets)
pace = long_run_pace(runs, compound_offsets, session_shift)
actual_quali = load_session_result(year, round_number, event, "Qualifying", 1.5)
official_grid = load_starting_grid(year, round_number, event) if actual_quali is not None else None
announced_drops = race_notes.get("grid_drops", {}) or {}
if official_grid is not None:
    known_grid = official_grid
elif actual_quali is not None and announced_drops:
    known_grid = apply_grid_drops(actual_quali, announced_drops)
else:
    known_grid = actual_quali
race = race_projection(quali, pace, grid_weight, known_grid)

# drivers whose start slot differs from where they qualified (penalties, pit-lane starts)
grid_changes = None
if official_grid is not None and actual_quali is not None:
    gc = actual_quali.rename(columns={"Position": "Qualified"}).merge(
        official_grid.rename(columns={"Position": "Starts"}), on="Driver", how="inner")
    gc = gc[gc["Qualified"] != gc["Starts"]].copy()
    if not gc.empty:
        gc["Qualified"] = gc["Qualified"].astype(int)
        gc["Starts"] = gc["Starts"].astype(int)
        gc["Change"] = gc["Starts"] - gc["Qualified"]
        gc = gc[gc["Change"] > 0]  # only penalised drivers; others just move up behind them
        grid_changes = gc.sort_values("Change", ascending=False) if not gc.empty else None
elif actual_quali is not None and announced_drops:
    gc = actual_quali.rename(columns={"Position": "Qualified"}).merge(
        known_grid.rename(columns={"Position": "Starts"}), on="Driver", how="inner")
    gc["Qualified"] = gc["Qualified"].astype(int)
    gc["Starts"] = gc["Starts"].astype(int)
    gc["Change"] = gc["Starts"] - gc["Qualified"]
    gc = gc[gc["Driver"].isin(announced_drops.keys())]
    grid_changes = gc.sort_values("Change", ascending=False) if not gc.empty else None

actual = load_session_result(year, round_number, event, "Race", 3)
if actual is not None:
    actual = actual.rename(columns={"Position": "ActualFinish"})
if race is not None and actual is not None:
    race = race.merge(actual, on="Driver", how="left")

if official_grid is not None:
    grid_label = "Grid (official, with penalties)"
    data_lines.append("Grid: official FIA grid, with penalties")
elif actual_quali is not None and announced_drops:
    grid_label = "Grid (quali + announced penalties)"
    data_lines.append("Grid: qualifying + announced penalties (official grid not out yet)")
elif actual_quali is not None:
    grid_label = "Grid (actual quali)"
    data_lines.append("Grid: actual qualifying (switches to the official grid once published)")
else:
    grid_label = "Projected grid"
    data_lines.append("Grid: projected from practice")
data_lines.append(f"Long runs: {min_long_run}+ laps · Grid weight: {grid_weight:.0%}")

with data_box:
    for line in data_lines:
        st.caption(line)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("#### 🛞 Pirelli's likely race strategy")
        if race_notes.get("strategy"):
            st.markdown(f"**{race_notes['strategy']}**")
            if race_notes.get("title"):
                st.caption(race_notes["title"])
            for line in race_notes.get("details", []):
                st.markdown(f"- {line}")
            plan = {c.upper() for c in race_notes.get("compounds", [])}
            if plan:
                if plan == selected_compounds:
                    st.caption("✓ Your tyre settings match this strategy.")
                else:
                    st.caption("Tip: choose \"Match Pirelli strategy\" under Tyre type in Settings.")
            src = race_notes.get("source", "Pirelli")
            if race_notes.get("url"):
                st.caption(f"Source: [{src}]({race_notes['url']})")
            else:
                st.caption(f"Source: {src}")
        else:
            st.write("Pirelli hasn't published a strategy call for this race yet.")
            st.caption("It usually appears after qualifying. The app checks Pirelli's press feed "
                       "automatically; tap Refresh data to check again.")
with right:
    with st.container(border=True):
        st.markdown("#### ⚠️ Grid penalties")
        if grid_changes is not None:
            gshow = grid_changes.copy()
            gshow["Change"] = gshow["Change"].map(lambda x: f"{x} place{'s' if x != 1 else ''} back")
            st.dataframe(gshow, hide_index=True, use_container_width=True)
            st.caption("From the official FIA grid." if official_grid is not None
                       else "From announced penalties (strategies.json). The official grid will replace this.")
        elif official_grid is not None:
            st.write("None. The official grid matches qualifying.")
        elif actual_quali is not None:
            st.write("Official grid not published yet.")
            st.caption("The FIA usually confirms it on race morning.")
        else:
            st.write("Shown after qualifying.")
        for note in race_notes.get("penalty_notes", []):
            st.caption(note)

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Podium", "⏱️ Qualifying", "🏎️ Race", "📈 Long runs"])

with tab1:
    if race is None or race.empty:
        st.write(f"No long runs of {min_long_run}+ laps found yet. Try lowering the minimum long-run length in Settings.")
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
    if actual_quali is not None:
        st.markdown("**Actual qualifying result**")
        aq = actual_quali.copy()
        aq["Position"] = aq["Position"].astype(int)
        if quali is not None:
            aq = aq.merge(quali[["Driver", "Team", "ProjectedGrid"]], on="Driver", how="left")
            aq["Off by"] = (aq["ProjectedGrid"] - aq["Position"]).abs().astype("Int64")
            aq["ProjectedGrid"] = aq["ProjectedGrid"].astype("Int64")
            aq = aq.rename(columns={"ProjectedGrid": "Predicted"})
            pred_top3 = set(quali.head(3)["Driver"])
            hits = len(pred_top3 & set(aq.head(3)["Driver"]))
            pole_ok = (not quali.empty) and quali.iloc[0]["Driver"] == aq.iloc[0]["Driver"]
            st.caption(f"Pole {'predicted ✓' if pole_ok else 'missed'} · "
                       f"{hits} of 3 front-runners predicted")
        st.dataframe(aq.rename(columns={"Position": "Actual"}), hide_index=True,
                     use_container_width=True)
        if quali is not None:
            with st.expander("Show the practice-based projection"):
                show = quali[["ProjectedGrid", "Driver", "Team", "QualiScore"]].copy()
                show["QualiScore"] = show["QualiScore"].round(3)
                st.dataframe(show.rename(columns={"QualiScore": "Avg gap to best (%)"}),
                             hide_index=True, use_container_width=True)
    elif quali is None:
        st.write("No qualifying projection available.")
    else:
        show = quali[["ProjectedGrid", "Driver", "Team", "QualiScore"]].copy()
        show["QualiScore"] = show["QualiScore"].round(3)
        show = show.rename(columns={"QualiScore": "Avg gap to best (%)"})
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption("Gaps are tyre-adjusted to medium-equivalent lap times (see Race tab for the values).")
        fig = px.bar(quali, x="Driver", y="QualiScore", color="Team",
                     color_discrete_map=TEAM_COLOURS,
                     title="Projected qualifying gap to fastest (%)")
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)

with tab3:
    if race is None or race.empty:
        st.write(f"No long runs of {min_long_run}+ laps found. Try lowering the minimum long-run length in Settings.")
    else:
        cols_shown = ["ProjectedFinish", "Driver", "Team", "ProjectedGrid"]
        if "ActualFinish" in race.columns:
            cols_shown.append("ActualFinish")
        cols_shown += ["RaceGapPct", "DegPerLap", "LongRunLaps", "Tyres"]
        show = race[cols_shown].copy()
        if "ActualFinish" in show.columns:
            show["ActualFinish"] = show["ActualFinish"].astype("Int64")
            show = show.rename(columns={"ActualFinish": "Actual finish"})
        show["ProjectedGrid"] = show["ProjectedGrid"].astype("Int64")
        show = show.rename(columns={"ProjectedGrid": grid_label})
        show["RaceGapPct"] = show["RaceGapPct"].round(3)
        show["DegPerLap"] = show["DegPerLap"].round(3)
        show = show.rename(columns={"RaceGapPct": "Long-run gap (%)",
                                    "DegPerLap": "Tyre deg (s/lap)",
                                    "LongRunLaps": "Long-run laps"})
        st.dataframe(show, hide_index=True, use_container_width=True)
        with st.expander("Tyre compound adjustment"):
            st.write("All lap times are converted to medium-tyre equivalents before comparing.")
            st.dataframe(pd.DataFrame([
                {"Compound": c.title(), "Seconds vs medium": f"{compound_offsets[c]:+.2f}",
                 "Source": compound_source[c]} for c in ("SOFT", "MEDIUM", "HARD")]),
                hide_index=True, use_container_width=True)
            st.caption("Measured = learned from drivers who ran more than one compound this weekend. "
                       "Typical = not enough data, so a standard value is used.")
            if session_shift:
                st.write("Track evolution correction (long runs, vs the first session used): " +
                         ", ".join(f"{k} {v:+.2f}s" for k, v in session_shift.items()))
        fig = px.scatter(race, x="RaceGapPct", y="DegPerLap", text="Driver",
                         color="Team", color_discrete_map=TEAM_COLOURS,
                         title="Long-run pace vs tyre degradation")
        fig.update_traces(textposition="top center")
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)

with tab4:
    if runlaps_all is None or runlaps_all.empty:
        st.write(f"No long runs of {min_long_run}+ laps found. Try lowering the minimum long-run length in Settings.")
    else:
        lr = runlaps_all.copy()
        # medium-equivalent, track-evolution-corrected lap times (what the model uses)
        lr["Adjusted"] = (lr["LapSeconds"] - lr["Compound"].map(offsets_all).fillna(0.0)
                          - lr["Session"].map(shift_all or {}).fillna(0.0))

        order = list(race["Driver"]) if race is not None and not race.empty else sorted(lr["Driver"].unique())
        order = [d for d in order if d in set(lr["Driver"])]
        picked = st.multiselect("Drivers", order, default=order[:6],
                                help="Defaults to the top 6 in the race projection.")
        c2, c3, c4 = st.columns([1.2, 2.2, 1.5])
        sess_opts = ["All"] + sorted(lr["Session"].unique())
        sess_pick = c2.selectbox("Session", sess_opts)
        tyre_opts = [c for c in ("SOFT", "MEDIUM", "HARD") if c in set(lr["Compound"])]
        tyre_default = [c for c in tyre_opts if c in selected_compounds] or tyre_opts
        tyres_pick = c3.multiselect("Tyres", tyre_opts, default=tyre_default,
                                    format_func=str.title,
                                    help="Starts with your Tyre type setting; change it here "
                                         "to look at other compounds without changing the predictions.")
        view = c4.radio("Lap times", ["Raw", "Adjusted"], horizontal=True,
                        help="Adjusted = converted to medium tyres and corrected for track "
                             "evolution, the way the model compares them.")
        y_col = "LapSeconds" if view == "Raw" else "Adjusted"

        sel = lr[lr["Driver"].isin(picked) & lr["Compound"].isin(tyres_pick)]
        if sess_pick != "All":
            sel = sel[sel["Session"] == sess_pick]
        if sel.empty:
            st.write("No long runs match those drivers, session and tyres.")
        else:
            dash = teammate_dash(set(zip(lr["Driver"], lr["Team"])))
            sel = sel.copy()
            sel["Line"] = sel["Driver"].map(dash)
            sel["Lap time"] = sel[y_col].map(lambda x: f"{int(x // 60)}:{x % 60:06.3f}")

            fig = px.line(sel, x="RunLap", y=y_col, color="Team", line_group="Run",
                          line_dash="Line", line_dash_map={"solid": "solid", "dash": "dash"},
                          symbol="Compound",
                          symbol_map={"SOFT": "circle", "MEDIUM": "diamond", "HARD": "square"},
                          markers=True, color_discrete_map=TEAM_COLOURS,
                          hover_name="Run",
                          hover_data={"Lap time": True, "TyreLife": True, "RunLap": False,
                                      y_col: False, "Line": False, "Team": True, "Compound": False},
                          labels={"RunLap": "Lap of the run", y_col: "Lap time (s)",
                                  "TyreLife": "Tyre age"},
                          title=f"Long-run lap times ({view.lower()})")
            seen_legend = set()
            for t in fig.data:
                t.name = t.name.split(",")[0]
                t.legendgroup = t.name
                t.showlegend = t.name not in seen_legend
                seen_legend.add(t.name)
            fig.update_layout(legend_title_text="Team", hovermode="closest")
            st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)
            st.caption("Solid line = first driver of each team, dashed = teammate. "
                       "● Soft  ◆ Medium  ■ Hard. Lower is faster.")

            med_order = sel.groupby("Driver")[y_col].median().sort_values().index.tolist()
            fig2 = px.box(sel, x="Driver", y=y_col, color="Team", points="all",
                          category_orders={"Driver": med_order},
                          color_discrete_map=TEAM_COLOURS,
                          labels={y_col: "Lap time (s)"},
                          title="Spread of long-run lap times (fastest median on the left)")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig2), use_container_width=True, theme=None)

            deg = runs_all[runs_all["Driver"].isin(picked) & runs_all["Compound"].isin(tyres_pick)]
            if sess_pick != "All":
                deg = deg[deg["Session"] == sess_pick]
            if not deg.empty:
                deg = deg.groupby(["Driver", "Team"])["DegPerLap"].mean().reset_index()
                deg = deg.sort_values("DegPerLap")
                fig3 = px.bar(deg, x="Driver", y="DegPerLap", color="Team",
                              color_discrete_map=TEAM_COLOURS,
                              labels={"DegPerLap": "Seconds lost per lap"},
                              title="Tyre degradation (lower is better)")
                fig3.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig3), use_container_width=True, theme=None)

st.caption("Built from FastF1 and OpenF1 data. Projections are estimates, not guarantees.")
