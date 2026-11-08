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
import threading
import time
from datetime import datetime

st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏆", layout="wide")
fastf1.Cache.enable_cache("/tmp")

# ---------- Model settings ----------
# Single-lap (qualifying) pace: FP3 is the closest thing to a quali sim, late FP2 next.
QUALI_WEIGHTS = {"FP1": 0.10, "FP2": 0.35, "FP3": 0.55}
QUALI_WEIGHTS_LEGACY = {"FP1": 0.20, "FP2": 0.35, "FP3": 0.45}
# Long-run (race) pace: FP2 is when everyone does their race sim on race fuel,
# so its long runs count fully; FP1/FP3 long runs are on unknown fuel and count less.
RACE_SESSION_WEIGHTS = {"FP1": 0.5, "FP2": 1.0, "FP3": 0.6}
FIRST_SEASON = 2021
GRID_WEIGHT_DEFAULT = 25        # percent, used when the circuit isn't in the table below
MIN_LONG_RUN_DEFAULT = 8        # laps; a real race sim, not a short run
FORM_WEIGHT_DEFAULT = 30        # percent of the final score that comes from recent results
FORM_RACES_DEFAULT = 4          # how many previous races feed the form prior
FORM_DECAY = 0.7                # each race further back counts 70% of the one after it
CORRECTION_RACES_DEFAULT = 4    # races used to learn each team's practice-vs-quali gap
CORRECTION_SHRINK = 1.0         # pseudo-races pulling the correction toward zero
CORRECTION_LIMIT = 0.6          # max correction, % of lap time
DEG_WEIGHT_DEFAULT = 1.0        # how hard tyre degradation counts in race pace (0 = ignore)
DEG_PROJECTION_LAPS = 6         # at weight 1.0, pace is projected from 5-lap-old tyres to ~11 laps
                                # old, the middle of a typical ~22-lap race stint
WET_GRID_DISCOUNT_DEFAULT = 50  # percent cut to the grid weight when qualifying was wet
WET_SHARE_THRESHOLD = 0.30      # a session is "wet" when this share of its laps are on inters/wets
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
SIM_RUNS = 4000                 # simulated races for the podium probabilities
SIM_NOISE = 0.6                 # race-day randomness, in score units (std devs)
DNF_RATE = 0.08                 # chance any one car fails to finish in a simulated race

# How much starting position matters, by circuit (percent). Street tracks and
# narrow circuits high, long straights and DRS zones low.
CIRCUIT_GRID_WEIGHT = [
    (("monaco", "monte carlo"), 55),
    (("budapest", "hungar"), 45),
    (("marina bay", "singapore"), 45),
    (("zandvoort", "dutch"), 45),
    (("imola", "emilia"), 40),
    (("barcelona", "catalunya", "spanish"), 35),
    (("madrid",), 35),
    (("suzuka", "japan"), 35),
    (("melbourne", "australia"), 35),
    (("yas", "abu dhabi"), 35),
    (("miami",), 30),
    (("mexico",), 30),
    (("lusail", "qatar"), 30),
    (("silverstone", "british"), 25),
    (("montr", "canad"), 25),
    (("shanghai", "chin"), 25),
    (("austin", "united states", "cota"), 20),
    (("monza", "italian"), 20),
    (("jeddah", "saudi"), 20),
    (("spielberg", "austria"), 20),
    (("baku", "azerbaijan"), 15),
    (("spa-francorchamps", "belgi"), 15),
    (("sakhir", "bahrain"), 15),
    (("las vegas",), 15),
    (("paulo", "interlagos", "brazil"), 15),
]

# Track evolution inside a session: how much faster the track gets per minute, in
# seconds, when this weekend's data isn't enough to measure it.
EVOLUTION_DEFAULT = {"FP1": -0.020, "FP2": -0.012, "FP3": -0.008}
EVOLUTION_LIMITS = (-0.06, 0.0)

# Typical lap-time difference vs the medium tyre, in seconds per lap.
# Used only when a weekend doesn't have enough mixed-compound runs to measure it.
COMPOUND_DEFAULTS = {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.4}
COMPOUND_LIMITS = {"SOFT": (-1.2, -0.1), "HARD": (0.1, 1.2)}
COMPOUND_ORDER = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
DRY_COMPOUNDS = set(COMPOUND_DEFAULTS)


def circuit_grid_weight(event):
    """Typical grid-position weight for this circuit, as a fraction."""
    text = f"{event.get('Location', '')} {event.get('EventName', '')} {event.get('Country', '')}".lower()
    for keys, pct in CIRCUIT_GRID_WEIGHT:
        if any(k in text for k in keys):
            return pct / 100
    return GRID_WEIGHT_DEFAULT / 100


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


def completed_rounds(year):
    """Schedule rows for races that have already finished this season, latest first."""
    sched = get_schedule(year)
    sched = sched[sched["RoundNumber"] > 0]
    now = pd.Timestamp.now(tz="UTC")
    done = []
    for _, ev in sched.iterrows():
        rd = session_date_for(ev, "Race")
        if rd is None:
            continue
        rd = rd.tz_localize("UTC") if rd.tzinfo is None else rd.tz_convert("UTC")
        if rd + pd.Timedelta(hours=3) < now:
            done.append(ev)
    return sorted(done, key=lambda e: -int(e["RoundNumber"]))


LAP_COLUMNS = ["Driver", "Team", "LapTime", "LapNumber", "Stint", "Compound",
               "TyreLife", "IsAccurate", "PitInTime", "PitOutTime", "LapSeconds", "TeamColor",
               "SessionTime"]


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
    # seconds into the session when the lap started (for track-evolution correction)
    try:
        if "LapStartTime" in laps.columns:
            laps["SessionTime"] = laps["LapStartTime"].dt.total_seconds()
        else:
            laps["SessionTime"] = (laps["Time"] - laps["LapTime"]).dt.total_seconds()
    except Exception:
        laps["SessionTime"] = np.nan
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
    sess_start = pd.Timestamp(best["date_start"]).tz_convert("UTC")

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

    if "date_start" in laps.columns:
        session_time = (pd.to_datetime(laps["date_start"], utc=True, errors="coerce") - sess_start).dt.total_seconds()
    else:
        session_time = pd.Series(np.nan, index=laps.index)
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
        "SessionTime": session_time,
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
    # Classified = finished the race (or qualified). Retirements still get a
    # Position in FastF1 but their ClassifiedPosition is a letter (R, D, ...).
    if "ClassifiedPosition" in sess.results.columns:
        res["Classified"] = pd.to_numeric(sess.results["ClassifiedPosition"], errors="coerce").notna()
    else:
        res["Classified"] = True
    res = res.dropna(subset=["Position"])
    if res.empty:
        raise ValueError(f"No {session_name} results yet")
    return res.rename(columns={"Abbreviation": "Driver"})[["Driver", "Position", "Classified"]]


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
    flags = pd.Series(False, index=res.index)
    for col in ("dnf", "dns", "dsq"):
        if col in res.columns:
            flags = flags | (res[col] == True)
    res["Classified"] = ~flags
    res = res.dropna(subset=["Position"])
    return res.rename(columns={"name_acronym": "Driver"})[["Driver", "Position", "Classified"]]


def load_session_result(year, round_number, event, session_name, done_after_hours, keep_classified=False):
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
            res = loader().sort_values("Position").reset_index(drop=True)
            if not keep_classified:
                res = res.drop(columns="Classified")
            return res
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


# ---------- Model ----------
def zscore(s):
    """Standardise a series so different measures (gap %, grid slot, form) can be blended
    while keeping how far apart the values are."""
    s = pd.Series(s, dtype=float)
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def _stint_runs(session_laps, min_laps=MIN_LONG_RUN_DEFAULT):
    """Every clean run of at least min_laps laps, with pace corrected to 5-lap-old tyres.
    Pace is the median lap of the run (one slow lap doesn't drag it). Degradation is only
    measured on runs of 6+ laps; shorter runs get the field's typical value later.
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
            if len(grp) >= 6:
                slope = np.polyfit(grp["TyreLife"].astype(float), grp["LapSeconds"], 1)[0]
                slope = max(min(slope, 0.3), -0.05)
            else:
                slope = np.nan
            avg_age = grp["TyreLife"].mean()
            corrected = grp["LapSeconds"].median() - (slope if np.isfinite(slope) else 0.0) * (avg_age - 5)
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
    runs = pd.DataFrame(rows)
    if not runs.empty and runs["DegPerLap"].notna().any():
        runs["DegPerLap"] = runs["DegPerLap"].fillna(runs["DegPerLap"].median())
    return runs, runlaps


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


def session_evolution(laps, offsets, default):
    """How much faster the track got per minute during one session (seconds, negative =
    faster). Measured from drivers who set two or more push laps at different times:
    pace = driver + rate x minutes, fitted with least squares. Falls back to a typical
    value when fewer than 4 drivers give usable pairs. Returns (rate, source)."""
    if laps is None or laps.empty or "SessionTime" not in laps.columns:
        return default, "typical"
    d = laps[laps["Compound"].astype(str).str.upper().isin(DRY_COMPOUNDS)
             & laps["LapSeconds"].notna() & laps["SessionTime"].notna()
             & laps["PitInTime"].isna() & laps["PitOutTime"].isna()].copy()
    if d.empty:
        return default, "typical"
    d["Adj"] = _medium_equivalent(d, offsets)
    best = d.groupby("Driver")["Adj"].transform("min")
    push = d[d["Adj"] <= best * 1.012].copy()  # push laps: within 1.2% of the driver's best
    counts = push.groupby("Driver").size()
    push = push[push["Driver"].isin(counts[counts >= 2].index)]
    if push["Driver"].nunique() < 4:
        return default, "typical"
    spread = push.groupby("Driver")["SessionTime"].agg(lambda x: x.max() - x.min())
    if spread.median() < 600:  # push laps need to be at least 10 minutes apart to say anything
        return default, "typical"
    y = push["Adj"] - push.groupby("Driver")["Adj"].transform("mean")
    x = (push["SessionTime"] - push.groupby("Driver")["SessionTime"].transform("mean")) / 60.0
    denom = float((x ** 2).sum())
    if denom == 0:
        return default, "typical"
    rate = float((x * y).sum() / denom)
    lo, hi = EVOLUTION_LIMITS
    return min(max(rate, lo), hi), "measured"


def session_best_gaps(laps, offsets=None, evolution_rate=None, softest_only=True):
    """Best lap per driver in one session, as % gap to the session's fastest.
    softest_only: use each driver's best lap on the softest tyre they ran (their quali
    sim), not a medium lap set on heavy fuel. evolution_rate: s/min the track gained;
    every lap is corrected to end-of-session conditions so early laps aren't penalised."""
    if laps is None or laps.empty:
        return None
    dry = laps[laps["Compound"].astype(str).str.upper().isin(DRY_COMPOUNDS)
               & laps["LapSeconds"].notna()].copy()
    if dry.empty:
        return None
    dry["AdjSeconds"] = _medium_equivalent(dry, offsets) if offsets else dry["LapSeconds"]
    if evolution_rate and "SessionTime" in dry.columns and dry["SessionTime"].notna().any():
        t_end = dry["SessionTime"].max()
        dry["AdjSeconds"] = dry["AdjSeconds"] + evolution_rate * (t_end - dry["SessionTime"]).fillna(0.0) / 60.0
    if softest_only:
        dry["_c"] = dry["Compound"].astype(str).str.upper().map(COMPOUND_ORDER)
        dry = dry[dry["_c"] == dry.groupby("Driver")["_c"].transform("min")]
    best = dry.loc[dry.groupby("Driver")["AdjSeconds"].idxmin(),
                   ["Driver", "Team", "AdjSeconds", "Compound"]]
    fastest = best["AdjSeconds"].min()
    best["GapPct"] = (best["AdjSeconds"] - fastest) / fastest * 100
    return best


def quali_projection(session_laps, offsets, weights, evolution=None, softest_only=True):
    """Practice-only single-lap ranking. Returns one row per driver with QualiScore
    (weighted % gap to the fastest) and the per-session gaps."""
    rows = []
    for name, laps in session_laps.items():
        rate = evolution.get(name) if evolution else None
        best = session_best_gaps(laps, offsets, rate, softest_only)
        if best is None:
            continue
        best = best.copy()
        best["Session"] = name
        rows.append(best)
    if not rows:
        return None
    allbest = pd.concat(rows)
    pivot = allbest.pivot_table(index=["Driver", "Team"], columns="Session",
                                values="GapPct").reset_index()
    score = pd.Series(0.0, index=pivot.index)
    total_w = pd.Series(0.0, index=pivot.index)
    for s_, w in weights.items():
        if s_ in pivot.columns:
            has = pivot[s_].notna()
            score[has] += pivot.loc[has, s_] * w
            total_w[has] += w
    pivot["QualiScore"] = score / total_w.replace(0, np.nan)
    pivot = pivot.dropna(subset=["QualiScore"]).sort_values("QualiScore").reset_index(drop=True)
    pivot["ProjectedGrid"] = range(1, len(pivot) + 1)
    return pivot


def wet_share(laps):
    """Share of a session's timed laps set on intermediate or wet tyres."""
    if laps is None or laps.empty:
        return 0.0
    timed = laps[laps["LapSeconds"].notna()]
    if timed.empty:
        return 0.0
    return float(timed["Compound"].astype(str).str.upper().isin(WET_COMPOUNDS).mean())


def qualifying_was_wet(year, round_number, event):
    """(True/False, share of quali laps on rain tyres). None if quali laps can't be loaded."""
    qlaps, _, _ = load_session_laps(year, round_number, "Qualifying", event)
    if qlaps is None or qlaps.empty:
        return None, None
    share = wet_share(qlaps)
    return share >= WET_SHARE_THRESHOLD, share


def _prior_rounds(year, round_number, n_races):
    """The n most recent completed races before this round, latest first. Reaches back
    into the previous season (at reduced weight) early in the year.
    Returns a list of (year, round, event row, weight scale)."""
    out = []
    try:
        sched = get_schedule(year)
        prior = sched[(sched["RoundNumber"] > 0) & (sched["RoundNumber"] < round_number)]
        for _, ev in prior.sort_values("RoundNumber", ascending=False).iterrows():
            out.append((year, int(ev["RoundNumber"]), ev, 1.0))
            if len(out) >= n_races:
                break
    except Exception:
        pass
    if len(out) < n_races and year - 1 >= FIRST_SEASON:
        try:
            prev = get_schedule(year - 1)
            prev = prev[prev["RoundNumber"] > 0].sort_values("RoundNumber", ascending=False)
            for _, ev in prev.iterrows():
                out.append((year - 1, int(ev["RoundNumber"]), ev, 0.6))
                if len(out) >= n_races:
                    break
        except Exception:
            pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def recent_form(year, round_number, n_races):
    """Each driver's weighted average finishing and qualifying position over the last
    n races before this round. Recent races count more (FORM_DECAY). Retirements don't
    count against a driver: a DNF is bad luck, not slow pace. Wet qualifying sessions are
    left out of the qualifying form.
    Returns a DataFrame (Driver, FormFinish, FormGrid, FormRaces) or None."""
    race_rows, quali_rows = [], []
    for i, (yr, rnd, ev, scale) in enumerate(_prior_rounds(year, round_number, n_races)):
        w = scale * (FORM_DECAY ** i)
        race = load_session_result(yr, rnd, ev, "Race", 3, keep_classified=True)
        if race is not None:
            race = race[race["Classified"]]
            race_rows.append(pd.DataFrame({"Driver": race["Driver"], "Pos": race["Position"], "W": w}))
        quali = load_session_result(yr, rnd, ev, "Qualifying", 1.5, keep_classified=True)
        if quali is not None:
            wet, _ = qualifying_was_wet(yr, rnd, ev)
            if not wet:  # a wet qualifying says little about one-lap pace; skip it
                quali_rows.append(pd.DataFrame({"Driver": quali["Driver"], "Pos": quali["Position"], "W": w}))
    if not race_rows and not quali_rows:
        return None

    def wavg(frames, col):
        if not frames:
            return pd.DataFrame(columns=["Driver", col, "N"])
        df = pd.concat(frames)
        df["WP"] = df["W"] * df["Pos"]
        g = df.groupby("Driver").agg(WP=("WP", "sum"), W=("W", "sum"), N=("Pos", "size")).reset_index()
        g[col] = g["WP"] / g["W"]
        return g[["Driver", col, "N"]]

    r = wavg(race_rows, "FormFinish").rename(columns={"N": "FormRaces"})
    q = wavg(quali_rows, "FormGrid").drop(columns="N")
    form = r.merge(q, on="Driver", how="outer")
    form["FormRaces"] = form["FormRaces"].fillna(0).astype(int)
    return form


@st.cache_data(ttl=3600, show_spinner=False)
def team_quali_correction(year, round_number, n_races):
    """How much each team's practice pace under- or over-states its qualifying pace,
    learned from the last n races: (practice gap %) - (actual qualifying gap %), averaged
    per team, centred on the field, recent races weighted more, shrunk toward zero and
    capped. Positive = the team looks slower in practice than it qualifies (sandbagging).
    Returns (dict team -> correction %, details DataFrame)."""
    deltas = []  # one row per driver per race
    used = 0
    for yr, rnd, ev, scale in _prior_rounds(year, round_number, n_races + 2):
        if used >= n_races:
            break
        practice = None
        for sname in ("Practice 3", "Practice 2"):
            laps, _, _ = load_session_laps(yr, rnd, sname, ev)
            if laps is not None and not laps.empty:
                practice = laps
                break
        if practice is None:
            continue  # sprint weekend or no data: skip this race
        qlaps, _, _ = load_session_laps(yr, rnd, "Qualifying", ev)
        if qlaps is None or qlaps.empty:
            continue
        if wet_share(qlaps) >= WET_SHARE_THRESHOLD or wet_share(practice) >= WET_SHARE_THRESHOLD:
            continue  # rain in either session: not a fair practice-vs-quali comparison
        p = session_best_gaps(practice, None, None, softest_only=True)
        q = session_best_gaps(qlaps[qlaps["IsAccurate"] == True], None, None, softest_only=False)
        if p is None or q is None:
            continue
        m = p[["Driver", "Team", "GapPct"]].merge(q[["Driver", "GapPct"]], on="Driver",
                                                    suffixes=("Practice", "Quali"))
        if len(m) < 10:
            continue
        m["Delta"] = m["GapPctPractice"] - m["GapPctQuali"]
        m["Delta"] = m["Delta"] - m["Delta"].mean()  # only the relative picture matters
        m["W"] = scale * (FORM_DECAY ** used)
        m["Race"] = f"{yr} R{rnd}"
        deltas.append(m[["Team", "Delta", "W", "Race"]])
        used += 1
    if not deltas:
        return {}, None
    df = pd.concat(deltas)
    df["WD"] = df["W"] * df["Delta"]
    g = df.groupby("Team").agg(WD=("WD", "sum"), W=("W", "sum"), Races=("Race", "nunique")).reset_index()
    g["Correction"] = (g["WD"] / (g["W"] + CORRECTION_SHRINK)).clip(-CORRECTION_LIMIT, CORRECTION_LIMIT)
    g = g.sort_values("Correction", ascending=False)
    return dict(zip(g["Team"], g["Correction"])), g[["Team", "Correction", "Races"]]


def quali_blend(quali, correction=None, form=None, form_weight=0.0):
    """Final qualifying projection: practice gap (team-corrected) blended with recent
    qualifying form, both standardised. Keeps QualiScore in gap % for display."""
    if quali is None:
        return None
    q = quali.copy()
    if correction:
        q["Correction"] = q["Team"].map(correction).fillna(0.0)
        q["QualiScore"] = q["QualiScore"] - q["Correction"]
        q["QualiScore"] = q["QualiScore"] - q["QualiScore"].min()
    q["QualiZ"] = zscore(q["QualiScore"])
    q["Blend"] = q["QualiZ"]
    if form is not None and form_weight > 0 and "FormGrid" in form.columns:
        q = q.merge(form[["Driver", "FormGrid"]], on="Driver", how="left")
        fz = zscore(q["FormGrid"].dropna())
        q["FormZ"] = np.nan
        q.loc[fz.index, "FormZ"] = fz.values
        q["FormZ"] = q["FormZ"].fillna(0.5)  # no recent results (rookie, new entry): a bit below average
        q["Blend"] = (1 - form_weight) * q["QualiZ"] + form_weight * q["FormZ"]
    q = q.sort_values("Blend").reset_index(drop=True)
    q["ProjectedGrid"] = range(1, len(q) + 1)
    return q


def long_run_pace(runs, offsets, session_shift=None, session_weights=None, deg_weight=0.0):
    """Race pace per driver from their best long runs, converted to medium-tyre, FP1
    conditions. Each run is weighted by laps x session weight (FP2 race sims count most),
    and the three best-weighted runs per driver are averaged. deg_weight then projects the
    pace deeper into a stint: a car that eats its tyres pays for it in the race, not just
    on lap 5. At 1.0 the pace is taken at the middle of a typical stint; 2.0 doubles that."""
    if runs is None or runs.empty:
        return None
    runs = runs.copy()
    runs["Pace"] = runs["Pace"] - runs["Compound"].map(offsets).fillna(0.0)
    if session_shift:
        runs["Pace"] = runs["Pace"] - runs["Session"].map(session_shift).fillna(0.0)
    sw = session_weights or {}
    runs["W"] = runs["Laps"].astype(float) * runs["Session"].map(sw).fillna(1.0)
    best_runs = runs.sort_values("W", ascending=False).groupby("Driver").head(3)

    rows = []
    for (drv, team), g in best_runs.groupby(["Driver", "Team"]):
        w = g["W"].to_numpy(dtype=float)
        deg = g["DegPerLap"].to_numpy(dtype=float)
        ok = np.isfinite(deg)
        rows.append({
            "Driver": drv, "Team": team,
            "RacePace": float(np.average(g["Pace"], weights=w)),
            "DegPerLap": float(np.average(deg[ok], weights=w[ok])) if ok.any() else np.nan,
            "LongRunLaps": int(g["Laps"].sum()),
            "Tyres": "/".join(sorted(set(g["Compound"]))),
        })
    out = pd.DataFrame(rows)
    out["Pace5"] = out["RacePace"]
    if deg_weight and out["DegPerLap"].notna().any():
        deg = out["DegPerLap"].fillna(out["DegPerLap"].median())
        out["DegPenalty"] = deg_weight * DEG_PROJECTION_LAPS * deg
        out["RacePace"] = out["Pace5"] + out["DegPenalty"]
    else:
        out["DegPenalty"] = 0.0
    fastest = out["RacePace"].min()
    out["RaceGapPct"] = (out["RacePace"] - fastest) / fastest * 100
    return out


def race_projection(quali, pace, grid_weight, known_grid=None, form=None, form_weight=0.0, legacy=False):
    """Blends long-run pace, grid position and recent race form. Uses the best grid
    available: official starting grid (with penalties) > real qualifying > projected.
    Pace enters as the actual % gap (standardised), so a car 0.8% off is treated very
    differently from one 0.05% off, even if they're neighbours in the order."""
    if pace is None or pace.empty:
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
    if legacy:  # the original rank-based blend, kept for the backtest comparison
        pace_rank = df["RaceGapPct"].rank()
        df["RaceScore"] = (1 - grid_weight) * pace_rank + grid_weight * df["ProjectedGrid"]
    else:
        df["PaceZ"] = zscore(df["RaceGapPct"])
        df["GridZ"] = zscore(df["ProjectedGrid"])
        df["PracticeScore"] = (1 - grid_weight) * df["PaceZ"] + grid_weight * df["GridZ"]
        df["RaceScore"] = df["PracticeScore"]
        if form is not None and form_weight > 0 and "FormFinish" in form.columns:
            df = df.merge(form[["Driver", "FormFinish", "FormRaces"]], on="Driver", how="left")
            fz = zscore(df["FormFinish"].dropna())
            df["FormZ"] = np.nan
            df.loc[fz.index, "FormZ"] = fz.values
            df["FormZ"] = df["FormZ"].fillna(0.5)
            df["RaceScore"] = (1 - form_weight) * df["PracticeScore"] + form_weight * df["FormZ"]
    df = df.sort_values("RaceScore").reset_index(drop=True)
    df["ProjectedFinish"] = df.index + 1
    return df


def simulate_race(race, n_runs=SIM_RUNS, noise=SIM_NOISE, dnf_rate=DNF_RATE, seed=7):
    """Runs the race thousands of times with random race-day noise and retirements,
    then counts how often each driver wins, makes the podium or finishes top 10."""
    if race is None or race.empty or "RaceScore" not in race.columns:
        return race
    rng = np.random.default_rng(seed)
    k = len(race)
    base = race["RaceScore"].to_numpy(dtype=float)[:, None]
    scores = base + rng.normal(0.0, noise, size=(k, n_runs))
    dnf = rng.random(size=(k, n_runs)) < dnf_rate
    scores = np.where(dnf, scores + 50.0, scores)
    ranks = scores.argsort(axis=0).argsort(axis=0) + 1
    out = race.copy()
    out["WinPct"] = (ranks == 1).mean(axis=1) * 100
    out["PodiumPct"] = (ranks <= 3).mean(axis=1) * 100
    out["Top10Pct"] = (ranks <= 10).mean(axis=1) * 100
    out["ExpectedFinish"] = ranks.mean(axis=1)
    return out


def score_prediction(pred, pred_col, actual, actual_col):
    """Scores a predicted order against the real one.
    Returns dict: top1 hit, top-3 hits (0-3), exact top-3 slots, mean abs position error,
    Spearman rank correlation."""
    if pred is None or actual is None or pred.empty or actual.empty:
        return None
    p = pred[["Driver", pred_col]].dropna()
    a = actual[["Driver", actual_col]].dropna()
    m = p.merge(a, on="Driver", how="inner")
    if len(m) < 5:
        return None
    pred_top3 = list(p.sort_values(pred_col)["Driver"].head(3))
    real_top3 = list(a.sort_values(actual_col)["Driver"].head(3))
    return {
        "Top1": pred_top3[0] == real_top3[0],
        "Top3Hits": len(set(pred_top3) & set(real_top3)),
        "Top3Exact": sum(x == y for x, y in zip(pred_top3, real_top3)),
        "MAE": float((m[pred_col] - m[actual_col]).abs().mean()),
        "Spearman": float(m[pred_col].corr(m[actual_col], method="spearman")),
        "Drivers": len(m),
    }


ALL_FP = {"FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"}

# Settings of the original model, for the backtest comparison
BASELINE_SETTINGS = {"legacy": True, "min_laps": 5, "grid_weight": 0.25, "form_weight": 0.0,
                     "form_races": 0, "correction": False, "correction_races": 0, "evolution": False,
                     "deg_weight": 0.0, "wet_quali": "dry", "wet_discount": 0.0}


def run_model(year, round_number, event, s):
    """Runs the whole pipeline for one race weekend with settings dict s and returns
    everything the UI and the backtest need. Shared so both score the same way."""
    legacy = s.get("legacy", False)
    sess_map = {k: v for k, v in ALL_FP.items() if k in s["fp"]}
    session_laps, load_errors, sources = {}, {}, {}
    for short, full in sess_map.items():
        laps, src, err = load_session_laps(year, round_number, full, event)
        session_laps[short] = laps
        if err:
            load_errors[short] = err
        else:
            sources[short] = src
    available = [k for k, v in session_laps.items() if v is not None and not v.empty]
    out = {"sess_map": sess_map, "load_errors": load_errors, "sources": sources,
           "available": available, "session_laps": session_laps, "quali": None, "race": None,
           "actual_quali": None, "actual": None, "no_compound_laps": False}
    if not available:
        return out

    session_laps_all = dict(session_laps)
    session_laps = {
        k: (v[v["Compound"].astype(str).str.upper().isin(s["compounds"])] if v is not None else None)
        for k, v in session_laps.items()
    }
    out["session_laps_all"] = session_laps_all
    out["session_laps"] = session_laps
    if all(v is None or v.empty for v in session_laps.values()):
        out["no_compound_laps"] = True
        return out

    runs, runlaps = _stint_runs(session_laps, s["min_laps"])
    runs_all, runlaps_all = _stint_runs(session_laps_all, s["min_laps"])
    offsets_all, _, shift_all = estimate_compound_offsets(runs_all)
    compound_offsets, compound_source, session_shift = estimate_compound_offsets(runs)

    evolution, evo_source = {}, {}
    if s["evolution"] and not legacy:
        for name, laps in session_laps.items():
            rate, src = session_evolution(laps, compound_offsets, EVOLUTION_DEFAULT.get(name, -0.01))
            evolution[name], evo_source[name] = rate, src

    quali_raw = quali_projection(session_laps, compound_offsets,
                                 QUALI_WEIGHTS_LEGACY if legacy else QUALI_WEIGHTS,
                                 evolution or None, softest_only=not legacy)
    form = None
    if s["form_weight"] > 0 and not legacy:
        try:
            form = recent_form(year, round_number, s["form_races"])
        except Exception:
            form = None
    correction, correction_details = {}, None
    if s["correction"] and not legacy:
        try:
            correction, correction_details = team_quali_correction(year, round_number, s["correction_races"])
        except Exception:
            correction, correction_details = {}, None
    quali = quali_raw if legacy else quali_blend(quali_raw, correction or None, form, s["form_weight"])
    pace = long_run_pace(runs, compound_offsets, session_shift, None if legacy else RACE_SESSION_WEIGHTS,
                         0.0 if legacy else s.get("deg_weight", 0.0))

    actual_quali = load_session_result(year, round_number, event, "Qualifying", 1.5)
    official_grid = load_starting_grid(year, round_number, event) if actual_quali is not None else None
    race_notes = load_race_notes(year, round_number)
    announced_drops = race_notes.get("grid_drops", {}) or {}
    if official_grid is not None:
        known_grid = official_grid
    elif actual_quali is not None and announced_drops:
        known_grid = apply_grid_drops(actual_quali, announced_drops)
    else:
        known_grid = actual_quali
    grid_weight = s["grid_weight"] if s["grid_weight"] is not None else circuit_grid_weight(event)
    # Wet qualifying: the grid says less about pace, so starting position counts for less
    wet_quali, wet_pct = False, None
    mode = s.get("wet_quali", "auto")
    if known_grid is not None and mode != "dry":
        if mode == "wet":
            wet_quali = True
        else:
            detected, wet_pct = qualifying_was_wet(year, round_number, event)
            wet_quali = bool(detected)
    grid_weight_dry = grid_weight
    if wet_quali:
        grid_weight = grid_weight * (1 - s.get("wet_discount", 0.0))
    race = race_projection(quali, pace, grid_weight, known_grid, form, s["form_weight"], legacy)
    if race is not None and not legacy:
        race = simulate_race(race)

    actual = load_session_result(year, round_number, event, "Race", 3)
    if actual is not None:
        actual = actual.rename(columns={"Position": "ActualFinish"})
    if race is not None and actual is not None:
        race = race.merge(actual, on="Driver", how="left")

    out.update({
        "runs": runs, "runlaps": runlaps, "runs_all": runs_all, "runlaps_all": runlaps_all,
        "offsets_all": offsets_all, "shift_all": shift_all,
        "compound_offsets": compound_offsets, "compound_source": compound_source,
        "session_shift": session_shift, "evolution": evolution, "evo_source": evo_source,
        "quali_raw": quali_raw, "quali": quali, "form": form,
        "correction": correction, "correction_details": correction_details,
        "pace": pace, "actual_quali": actual_quali, "official_grid": official_grid,
        "announced_drops": announced_drops, "known_grid": known_grid,
        "grid_weight": grid_weight, "grid_weight_dry": grid_weight_dry,
        "wet_quali": wet_quali, "wet_pct": wet_pct, "race": race, "actual": actual,
    })
    return out


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
    /* Force the light look even if the browser or Streamlit picks dark mode */
    :root {{ color-scheme: light; }}
    [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"], [data-testid="stBottomBlockContainer"] {{
        background-color: #ffffff !important; color: {p['pine']} !important;
    }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background-color: {p['pine']} !important;
    }}
    [data-testid="stApp"] p, [data-testid="stApp"] li, [data-testid="stApp"] label,
    [data-testid="stApp"] span, [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {{
        color: {p['pine']} !important;
    }}
    [data-testid="stCaptionContainer"] {{ opacity: .8; }}
    /* inputs, dropdowns, multiselects */
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="input"] input,
    [data-testid="stNumberInput"] input, [data-baseweb="popover"] ul, [data-baseweb="menu"] {{
        background-color: #ffffff !important; color: {p['pine']} !important;
        border-color: {p['leaf']} !important;
    }}
    [data-baseweb="popover"] li {{ color: {p['pine']} !important; }}
    [data-baseweb="popover"] li:hover {{ background-color: {p['mint']} !important; }}
    [data-baseweb="tag"] {{ background-color: {p['forest']} !important; }}
    [data-baseweb="tag"] span {{ color: #ffffff !important; }}
    /* tables: light mint rows, light-green header, dark text */
    [data-testid="stTable"] {{ overflow-x: auto; }}
    [data-testid="stTable"] table {{
        width: 100%; border-collapse: collapse; background: #ffffff !important;
        border: 1px solid {p['leaf']} !important; border-radius: 8px; overflow: hidden;
        font-size: .92rem;
    }}
    [data-testid="stTable"] th {{
        background: {p['mint']} !important; color: {p['pine']} !important;
        font-weight: 700 !important; text-align: left !important;
        border-bottom: 2px solid {p['leaf']} !important; padding: .45rem .6rem !important;
    }}
    [data-testid="stTable"] td {{
        background: #ffffff !important; color: {p['pine']} !important;
        border-bottom: 1px solid rgba(104,186,127,0.35) !important; padding: .4rem .6rem !important;
    }}
    [data-testid="stTable"] tr:nth-child(even) td {{ background: #F3FFF6 !important; }}
    [data-testid="stTable"] tr:hover td {{ background: {p['mint']} !important; }}
    /* code blocks and alerts */
    [data-testid="stCode"], pre, code {{ background-color: {p['mint']} !important; color: {p['pine']} !important; }}
    [data-testid="stAlert"] {{ background-color: rgba(207,255,220,0.6) !important; }}
    [data-testid="stAlert"] p {{ color: {p['pine']} !important; }}
    h1, h2, h3, h4 {{ color: {p['pine']} !important; }}
    /* dark green Settings pane: light text, white inputs */
    [data-testid="stApp"] [data-testid="stSidebar"] h1,
    [data-testid="stApp"] [data-testid="stSidebar"] h2,
    [data-testid="stApp"] [data-testid="stSidebar"] h3,
    [data-testid="stApp"] [data-testid="stSidebar"] p,
    [data-testid="stApp"] [data-testid="stSidebar"] li,
    [data-testid="stApp"] [data-testid="stSidebar"] label,
    [data-testid="stApp"] [data-testid="stSidebar"] span,
    [data-testid="stApp"] [data-testid="stSidebar"] summary,
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        color: {p['mint']} !important;
    }}
    [data-testid="stApp"] [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stApp"] [data-testid="stSidebar"] [data-baseweb="input"] * {{
        color: {p['pine']} !important;
    }}
    /* status boxes inside the dark pane: dark tint, light text */
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stAlert"],
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stAlert"] > div {{
        background-color: rgba(37, 61, 44, 0.65) !important;
        border: 1px solid {p['leaf']} !important;
    }}
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stAlert"] *,
    [data-testid="stApp"] [data-testid="stSidebar"] code {{
        color: {p['mint']} !important; background-color: transparent !important;
    }}
    [data-testid="stApp"] [data-testid="stSidebar"] a {{ color: {p['mint']} !important; text-decoration: underline; }}
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stExpander"] svg {{
        color: {p['mint']} !important; fill: {p['mint']} !important;
    }}
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ opacity: 1; }}
    [data-testid="stApp"] [data-testid="stSidebar"] [data-testid="stExpander"] details {{
        background: {p['forest']} !important; border-color: {p['leaf']} !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{ background: {p['leaf']}; color: {p['pine']}; }}
    [data-testid="stSidebar"] .stButton > button p {{ color: {p['pine']} !important; }}
    [data-testid="stSidebar"] .stButton > button:hover {{ background: {p['mint']}; }}
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
    [data-testid="stSidebar"] [data-testid="stThumbValue"] {{ color: {p['mint']} !important; }}
    [data-testid="stSidebar"] [role="radio"] > div:first-child {{ border-color: {p['mint']} !important; }}
    [data-testid="stSidebar"] [role="checkbox"] {{ border-color: {p['mint']} !important; }}
    [data-testid="stSidebarCollapseButton"] svg, [data-testid="stSidebarHeader"] svg {{ fill: {p['mint']}; color: {p['mint']}; }}
    h1 {{ border-bottom: 3px solid {p['leaf']}; padding-bottom: .3rem; }}
    [data-testid="stSidebar"] {{ border-right: 2px solid {p['leaf']}; }}
    [data-testid="stSidebar"] h1 {{ border-bottom: none; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {p['leaf']} !important;
        background: rgba(207, 255, 220, 0.35);
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {p['forest']} !important;
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
    .stTabs [aria-selected="true"] {{ background: {p['leaf']}; }}
    .stTabs [aria-selected="true"] p {{ color: {p['pine']} !important; font-weight: 700; }}
    /* tags in multiselects (forest green chips) */
    [data-baseweb="tag"] * {{ color: #ffffff !important; fill: #ffffff !important; }}
    /* main-page buttons are forest green: keep their text white */
    [data-testid="stMain"] .stButton > button p {{ color: #ffffff !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {p['forest']} !important; }}
    .stButton > button {{
        background: {p['forest']}; color: #ffffff; border: none; font-weight: 700;
    }}
    .stButton > button:hover {{ background: {p['leaf']}; color: #ffffff; }}
    [data-testid="stExpander"] details {{ border-color: {p['leaf']} !important; }}
    [data-testid="stProgress"] > div > div > div > div {{ background-color: {p['forest']} !important; }}
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


def show_table(df):
    """Light-styled HTML table (Streamlit's interactive grid can't be recoloured)."""
    sty = df.style.hide(axis="index").format(precision=3, na_rep="–")
    st.table(sty)


def result_banner(text, icon="🎯"):
    """Forest-green score box with light text, for how the prediction did."""
    p = PALETTE
    st.markdown(f"""
    <div style="background:{p['forest']};border:1px solid {p['leaf']};border-radius:12px;
                padding:.7rem 1rem;margin:.4rem 0 .8rem 0;color:#ffffff;font-size:1.05rem;
                font-weight:700;">
      <span style="margin-right:.5rem;">{icon}</span>{text}
    </div>""", unsafe_allow_html=True)


def podium_card(slot, medal, driver, team, note=None, good=None):
    """Forest-green podium card with light text and the team's colour as an accent."""
    p = PALETTE
    accent = TEAM_COLOURS.get(team, p["leaf"])
    badge = ""
    if note:
        bc = p["mint"] if good else "#FFD6D6"
        badge = (f'<div style="margin-top:.35rem;font-size:.85rem;font-weight:700;color:{bc};">'
                 f'{note}</div>')
    team_line = f'<div style="font-size:.95rem;color:{p["mint"]};opacity:.95;">{team}</div>' if team else ""
    slot.markdown(f"""
    <div style="background:{p['forest']};border:1px solid {p['leaf']};border-left:8px solid {accent};
                border-radius:12px;padding:.8rem 1rem;margin-bottom:.5rem;">
      <div style="font-size:.95rem;color:{p['mint']};font-weight:600;">{medal}</div>
      <div style="font-size:2rem;font-weight:800;color:#ffffff;line-height:1.15;">{driver}</div>
      {team_line}{badge}
    </div>""", unsafe_allow_html=True)


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
round_number = int(event["RoundNumber"])
race_notes = load_race_notes(year, round_number)
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
                           help="Only the selected sessions feed the projections. Qualifying pace "
                                "leans on FP3, race pace on FP2's long runs.")
selected_fp = FP_CHOICES[fp_pick]

min_long_run = st.sidebar.slider(
    "Minimum long-run length (laps)", 3, 15, MIN_LONG_RUN_DEFAULT,
    help="A run on one set of tyres needs at least this many clean laps to count as a long run. "
         "8+ is a real race sim; lower it if too few drivers show up in the race projection.")

auto_gw = circuit_grid_weight(event)
with st.sidebar.expander("Model tuning", expanded=False):
    grid_weight = st.slider(
        "Grid position weight", 0, 60, int(round(auto_gw * 100)), step=5, format="%d%%",
        key=f"gw_{year}_{round_number}",
        help="How much starting position counts in the race prediction. Starts at the typical "
             "value for this circuit (high at Monaco and Hungary, low at Spa and Baku). "
             "0% = practice long-run pace only.") / 100
    form_weight = st.slider(
        "Recent form weight", 0, 60, FORM_WEIGHT_DEFAULT, step=5, format="%d%%",
        help="How much the last few races' results count versus this weekend's practice data. "
             "Practice is noisy; form steadies it. 0% = practice only.") / 100
    history_races = st.slider(
        "Races of history to use", 2, 8, FORM_RACES_DEFAULT,
        help="Previous races used for recent form and for the team practice correction. "
             "The most recent race counts most. Early in the season this reaches back into last year.")
    deg_weight = st.slider(
        "Tyre degradation weight", 0.0, 3.0, DEG_WEIGHT_DEFAULT, step=0.25,
        help="How much a car's long-run degradation counts in its race pace. At 1.0 the pace is "
             "projected to the middle of a typical stint, so a car losing 0.1 s/lap more than another "
             "gives up about 0.6 s. At 2.0 that doubles. 0 = compare 5-lap-old pace only.")
    wet_mode_label = st.radio(
        "Qualifying conditions", ["Auto-detect", "Dry", "Wet"], index=0, horizontal=True,
        help="Auto-detect looks at how many qualifying laps were on inters or wets. "
             "In a wet qualifying the grid says less about pace, so its weight is cut.")
    wet_mode = {"Auto-detect": "auto", "Dry": "dry", "Wet": "wet"}[wet_mode_label]
    wet_discount = st.slider(
        "Wet qualifying grid discount", 0, 100, WET_GRID_DISCOUNT_DEFAULT, step=10, format="%d%%",
        help="How much to cut the grid weight when qualifying was wet. 50% halves it; 100% ignores "
             "the grid and runs on practice pace and form only.") / 100
    use_correction = st.checkbox(
        "Team practice correction", True,
        help="Learns which teams look slower in practice than they qualify (sandbagging) from the "
             "last few races and corrects for it. Slower first load: it downloads FP3 and "
             "qualifying laps for those races.")
    use_evolution = st.checkbox(
        "Track evolution correction", True,
        help="Corrects every lap to end-of-session grip, so a quick lap set early in a session "
             "isn't punished for a green track.")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

settings = {"fp": selected_fp, "compounds": selected_compounds, "min_laps": min_long_run,
            "grid_weight": grid_weight, "form_weight": form_weight, "form_races": history_races,
            "correction": use_correction, "correction_races": history_races,
            "evolution": use_evolution, "deg_weight": deg_weight,
            "wet_quali": wet_mode, "wet_discount": wet_discount, "legacy": False}

set_track_background(event, year)
circuit = event["Location"]
st.subheader(f"{event['EventName']} – {year}, Round {round_number}")
st.caption(f"{circuit}, {event['Country']}")

with st.spinner("Loading practice data and recent results..."):
    m = run_model(year, round_number, event, settings)

sess_map, load_errors, sources, available = m["sess_map"], m["load_errors"], m["sources"], m["available"]

with st.sidebar.expander("Diagnostics", expanded=not available):
    st.write(f"FastF1 version: {fastf1.__version__}")
    try:
        cfg_base = st.get_option("theme.base")
        cfg_bg = st.get_option("theme.backgroundColor")
    except Exception:
        cfg_base, cfg_bg = None, None
    if cfg_bg:
        st.success(f"Theme config found: base={cfg_base}, background={cfg_bg}")
    else:
        st.warning("Theme config file NOT found. Check .streamlit/config.toml sits next to app.py.")
    st.write(f"Event picked: {event['EventName']} (round {round_number})")
    for short in sess_map:
        if short in load_errors:
            st.error(f"{short}: {load_errors[short]}")
        else:
            st.success(f"{short}: {len(m['session_laps'][short])} laps from {sources[short]}")
    if m.get("evolution"):
        st.write("Track evolution (s/min): " + ", ".join(
            f"{k} {v:+.3f} ({m['evo_source'][k]})" for k, v in m["evolution"].items()))
    if m.get("form") is not None:
        st.write(f"Recent form: {len(m['form'])} drivers with results")
    elif settings["form_weight"] > 0 and available:
        st.warning("Recent form: no previous results could be loaded, so form is not being used.")
    if settings["correction"] and available:
        if m.get("correction"):
            st.write(f"Team practice correction: {len(m['correction'])} teams")
        else:
            st.warning("Team practice correction: could not be learned (no previous practice + qualifying laps), so none applied.")

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
if m["no_compound_laps"]:
    st.warning("No laps on the selected tyre type in these sessions. Try another option under Tyre type.")
    st.stop()
data_lines.append("Tyres: " + ", ".join(c.title() for c in ("SOFT", "MEDIUM", "HARD")
                                          if c in selected_compounds))

runs_all, runlaps_all = m["runs_all"], m["runlaps_all"]
offsets_all, shift_all = m["offsets_all"], m["shift_all"]
compound_offsets, compound_source, session_shift = m["compound_offsets"], m["compound_source"], m["session_shift"]
evolution = m["evolution"]
quali, race, pace = m["quali"], m["race"], m["pace"]
actual_quali, official_grid, known_grid = m["actual_quali"], m["official_grid"], m["known_grid"]
announced_drops, actual = m["announced_drops"], m["actual"]
form, correction, correction_details = m["form"], m["correction"], m["correction_details"]
grid_weight = m["grid_weight"]
wet_quali, wet_pct = m["wet_quali"], m["wet_pct"]
TEAM_COLOURS = team_colour_map(m["session_laps_all"])

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
gw_note = " (circuit default)" if abs(m["grid_weight_dry"] - auto_gw) < 1e-9 else ""
data_lines.append(f"Long runs: {min_long_run}+ laps, FP2 counts most · Tyre deg weight: {deg_weight:g}")
if wet_quali:
    shown_pct = f", {wet_pct:.0%} of laps on rain tyres" if wet_pct is not None else ""
    data_lines.append(f"Grid weight: {grid_weight:.0%} (wet qualifying{shown_pct}; cut from {m['grid_weight_dry']:.0%})")
else:
    data_lines.append(f"Grid weight: {grid_weight:.0%}{gw_note}")
if form is not None and form_weight > 0:
    data_lines.append(f"Recent form: last {history_races} races at {form_weight:.0%}")
else:
    data_lines.append("Recent form: off")
if correction:
    data_lines.append(f"Team practice correction: on ({len(correction)} teams)")
else:
    data_lines.append("Team practice correction: off")
if evolution:
    measured = [k for k, v in m["evo_source"].items() if v == "measured"]
    data_lines.append("Track evolution: " + (f"measured in {', '.join(measured)}" if measured else "typical values"))

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
            show_table(gshow)
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Podium", "⏱️ Qualifying", "🏎️ Race", "📈 Long runs", "📊 Backtest"])

with tab1:
    if race is None or race.empty:
        st.write(f"No long runs of {min_long_run}+ laps found yet. Try lowering the minimum long-run length in Settings.")
    else:
        top3 = race.head(3)
        cols = st.columns(3)
        medals = ["🥇 P1", "🥈 P2", "🥉 P3"]
        for c, mdl, (_, r) in zip(cols, medals, top3.iterrows()):
            note = f"{r['PodiumPct']:.0f}% podium · {r['WinPct']:.0f}% win" if "PodiumPct" in race.columns else None
            podium_card(c, mdl, r["Driver"], r["Team"], note, True)
        st.caption(f"Predicted podium from {SIM_RUNS:,} simulated races. Grid weight {grid_weight:.0%}"
                   f"{' (cut for wet qualifying)' if wet_quali else ''}, recent form {form_weight:.0%}, "
                   f"tyre deg weight {deg_weight:g}.")
        if actual is not None:
            st.markdown("**Actual podium**")
            real = actual.sort_values("ActualFinish").head(3)
            acols = st.columns(3)
            pred3 = set(top3["Driver"])
            team_of = dict(zip(race["Driver"], race["Team"]))
            for c, mdl, (_, r) in zip(acols, medals, real.iterrows()):
                ok = r["Driver"] in pred3
                podium_card(c, mdl, r["Driver"], team_of.get(r["Driver"], ""),
                            "✓ predicted" if ok else "✗ missed", ok)
            hits = len(pred3 & set(real["Driver"]))
            exact = sum(a == b for a, b in zip(top3["Driver"], real["Driver"]))
            result_banner(f"{hits} of 3 podium drivers predicted, "
                          f"<span style='color:{PALETTE['mint']};'>{exact} in the exact position</span>.")
        if "PodiumPct" in race.columns:
            st.markdown("**Podium odds**")
            odds = race[["Driver", "Team", "WinPct", "PodiumPct", "Top10Pct", "ExpectedFinish"]].head(10).copy()
            for col in ("WinPct", "PodiumPct", "Top10Pct"):
                odds[col] = odds[col].round(0).astype(int).astype(str) + "%"
            odds["ExpectedFinish"] = odds["ExpectedFinish"].round(1)
            show_table(odds.rename(columns={"WinPct": "Win", "PodiumPct": "Podium", "Top10Pct": "Top 10",
                                            "ExpectedFinish": "Avg finish"}))
            fig = px.bar(race.head(10), x="Driver", y="PodiumPct", color="Team",
                         color_discrete_map=TEAM_COLOURS, labels={"PodiumPct": "Podium chance (%)"},
                         title="Podium chance")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)

with tab2:
    def quali_table(q):
        cols = ["ProjectedGrid", "Driver", "Team", "QualiScore"]
        names = {"QualiScore": "Practice gap (%)"}
        if "Correction" in q.columns and (q["Correction"] != 0).any():
            cols.append("Correction")
            names["Correction"] = "Team correction (%)"
        if "FormGrid" in q.columns and q["FormGrid"].notna().any():
            cols.append("FormGrid")
            names["FormGrid"] = "Form avg quali"
        show = q[cols].copy()
        show["QualiScore"] = show["QualiScore"].round(3)
        if "Correction" in show.columns:
            show["Correction"] = show["Correction"].round(2)
        if "FormGrid" in show.columns:
            show["FormGrid"] = show["FormGrid"].round(1)
        return show.rename(columns=names)

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
            result_banner(f"Pole {'predicted ✓' if pole_ok else 'missed'} · "
                          f"<span style='color:{PALETTE['mint']};'>{hits} of 3 front-runners predicted</span>")
        show_table(aq.rename(columns={"Position": "Actual"}))
        if quali is not None:
            with st.expander("Show the practice-based projection"):
                show_table(quali_table(quali))
    elif quali is None:
        st.write("No qualifying projection available.")
    else:
        show_table(quali_table(quali))
        st.caption("Practice gap = each driver's best lap on their softest tyre, converted to medium-equivalent "
                   "and corrected for track evolution, weighted FP3 > FP2 > FP1. Order blends that with "
                   "recent qualifying form.")
        fig = px.bar(quali, x="Driver", y="QualiScore", color="Team",
                     color_discrete_map=TEAM_COLOURS,
                     title="Projected qualifying gap to fastest (%)")
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)
    if correction_details is not None and correction:
        with st.expander("Team practice correction"):
            st.write("Positive = the team has been qualifying better than its practice pace suggested "
                     f"over the last {history_races} races, so its practice gap is reduced by this much.")
            cd = correction_details.copy()
            cd["Correction"] = cd["Correction"].round(2)
            show_table(cd.rename(columns={"Correction": "Correction (% of lap)", "Races": "Races used"}))

with tab3:
    if race is None or race.empty:
        st.write(f"No long runs of {min_long_run}+ laps found. Try lowering the minimum long-run length in Settings.")
    else:
        cols_shown = ["ProjectedFinish", "Driver", "Team", "ProjectedGrid"]
        if "ActualFinish" in race.columns:
            cols_shown.append("ActualFinish")
        cols_shown += ["RaceGapPct"]
        if "FormFinish" in race.columns:
            cols_shown.append("FormFinish")
        if "PodiumPct" in race.columns:
            cols_shown.append("PodiumPct")
        cols_shown += ["DegPerLap", "LongRunLaps", "Tyres"]
        show = race[cols_shown].copy()
        if "ActualFinish" in show.columns:
            show["ActualFinish"] = show["ActualFinish"].astype("Int64")
            show = show.rename(columns={"ActualFinish": "Actual finish"})
        show["ProjectedGrid"] = show["ProjectedGrid"].astype("Int64")
        show = show.rename(columns={"ProjectedGrid": grid_label})
        show["RaceGapPct"] = show["RaceGapPct"].round(3)
        show["DegPerLap"] = show["DegPerLap"].round(3)
        if "FormFinish" in show.columns:
            show["FormFinish"] = show["FormFinish"].round(1)
        if "PodiumPct" in show.columns:
            show["PodiumPct"] = show["PodiumPct"].round(0).astype(int).astype(str) + "%"
        show = show.rename(columns={"RaceGapPct": "Race pace gap (%)",
                                    "FormFinish": "Form avg finish",
                                    "PodiumPct": "Podium",
                                    "DegPerLap": "Tyre deg (s/lap)",
                                    "LongRunLaps": "Long-run laps"})
        show_table(show)
        wet_note = " Qualifying was wet, so the grid weight was cut." if wet_quali else ""
        st.caption("Race pace gap = long-run pace with tyre degradation projected into the stint "
                   f"(deg weight {deg_weight:g}), blended with grid position at {grid_weight:.0%} and recent "
                   f"race form at {form_weight:.0%}. FP2 long runs count fully; FP1/FP3 long runs count "
                   f"about half, since fuel loads there are unknown.{wet_note}")
        with st.expander("Tyre compound and track adjustments"):
            st.write("All lap times are converted to medium-tyre equivalents before comparing.")
            show_table(pd.DataFrame([
                {"Compound": c.title(), "Seconds vs medium": f"{compound_offsets[c]:+.2f}",
                 "Source": compound_source[c]} for c in ("SOFT", "MEDIUM", "HARD")]))
            st.caption("Measured = learned from drivers who ran more than one compound this weekend. "
                       "Typical = not enough data, so a standard value is used.")
            if session_shift:
                st.write("Session-to-session track improvement (long runs, vs the first session used): " +
                         ", ".join(f"{k} {v:+.2f}s" for k, v in session_shift.items()))
            if evolution:
                st.write("Within-session track evolution (seconds gained per minute): " +
                         ", ".join(f"{k} {v:+.3f} ({m['evo_source'][k]})" for k, v in evolution.items()))
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
            if not deg.empty and deg["DegPerLap"].notna().any():
                deg = deg.groupby(["Driver", "Team"])["DegPerLap"].mean().reset_index()
                deg = deg.sort_values("DegPerLap")
                fig3 = px.bar(deg, x="Driver", y="DegPerLap", color="Team",
                              color_discrete_map=TEAM_COLOURS,
                              labels={"DegPerLap": "Seconds lost per lap"},
                              title="Tyre degradation (lower is better)")
                fig3.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig3), use_container_width=True, theme=None)

# ---------- Backtest storage: results survive closing the app and resume if a run is cut off ----------
BACKTEST_DIR = Path("/tmp/f1_backtest")


def backtest_signature(s):
    """Short text describing the settings a backtest was run with."""
    parts = [f"fp={'+'.join(s['fp'])}", f"tyres={'+'.join(sorted(s['compounds']))}", f"runs={s['min_laps']}",
             "grid=auto" if s["grid_weight"] is None else f"grid={s['grid_weight']:.0%}",
             f"form={s['form_weight']:.0%}x{s['form_races']}", f"corr={'on' if s['correction'] else 'off'}",
             f"evo={'on' if s['evolution'] else 'off'}", f"deg={s['deg_weight']:g}",
             f"wet={s['wet_quali']}/{s['wet_discount']:.0%}"]
    return " · ".join(parts)


def backtest_path(year):
    return BACKTEST_DIR / f"backtest_{year}.json"


def load_backtest(year):
    try:
        return json.loads(backtest_path(year).read_text(encoding="utf-8"))
    except Exception:
        return None


def save_backtest(year, data):
    try:
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        backtest_path(year).write_text(json.dumps(data, default=str), encoding="utf-8")
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def backtest_jobs():
    """Status of background backtests, shared by every open browser tab."""
    return {}


def _score_one(year, rnd, ev, label_, cfg):
    row = {"Model": label_, "Round": rnd, "Race": ev["EventName"]}
    try:
        mm = run_model(year, rnd, ev, cfg)
        qs = score_prediction(mm["quali"], "ProjectedGrid", mm["actual_quali"], "Position")
        rs = score_prediction(mm["race"], "ProjectedFinish", mm["actual"], "ActualFinish")
        if qs:
            row.update({"Pole": qs["Top1"], "Q top3 hits": qs["Top3Hits"],
                        "Q avg error": qs["MAE"], "Q correlation": qs["Spearman"]})
        if rs:
            row.update({"Winner": rs["Top1"], "Podium hits": rs["Top3Hits"],
                        "Podium exact": rs["Top3Exact"], "R avg error": rs["MAE"],
                        "R correlation": rs["Spearman"]})
        if not qs and not rs:
            row["Note"] = "no data"
    except Exception as e:
        row["Note"] = f"{type(e).__name__}: {e}"
    return row


def _backtest_worker(year, queue, sig, n_test, registry):
    """Runs in a background thread so the browser can disconnect and the work carries on.
    Saves after every race."""
    j = registry[year]
    try:
        for label_, rnd, ev, cfg in queue:
            if j.get("stop"):
                break
            j["current"] = f"{ev['EventName']}: {label_}"
            row = _score_one(year, rnd, ev, label_, cfg)
            data = load_backtest(year) or {"signature": sig, "rows": []}
            rows = [r for r in data.get("rows", []) if not (r["Model"] == label_ and r["Round"] == rnd)] + [row]
            j["done"] += 1
            save_backtest(year, {"signature": sig, "rows": rows, "races": n_test,
                                 "complete": j["done"] >= j["total"],
                                 "updated": datetime.now().strftime("%d %b %Y %H:%M")})
        j["status"] = "stopped" if j.get("stop") else "done"
    except Exception as e:
        j["status"] = "error"
        j["error"] = f"{type(e).__name__}: {e}"


with tab5:
    st.markdown("#### How accurate is the model?")
    st.write("Re-runs the prediction for past races this season and scores it against what actually "
             "happened. Qualifying is scored from practice only. The race is scored the way the app "
             "works after qualifying: real grid, practice long runs, recent form.")
    done_rounds = completed_rounds(year)
    if not done_rounds:
        st.write("No completed races this season yet.")
    else:
        test_settings = dict(settings)
        test_settings["grid_weight"] = None if gw_note else settings["grid_weight"]  # auto-by-circuit unless overridden
        sig = backtest_signature(test_settings)
        saved = load_backtest(year)

        n_test = st.slider("Races to test (most recent first)", 1, len(done_rounds), min(5, len(done_rounds)))
        compare = st.checkbox("Also score the previous model (rank blend, 25% grid, 5-lap runs, no form, deg or team correction)", True)
        st.caption("Results only change when you press Run. They're saved as each race finishes, so if the "
                   "app closes or times out part-way, press Run again and it picks up where it stopped. "
                   "Each race's form and team correction only use races before it, so nothing leaks.")
        registry = backtest_jobs()
        job = registry.get(year)
        running = bool(job and job.get("status") == "running")
        c_run, c_stop, c_clear = st.columns([1, 1, 1])
        run_clicked = c_run.button("Run backtest", disabled=running)
        if running and c_stop.button("Stop"):
            job["stop"] = True
        if c_clear.button("Clear saved results", disabled=running):
            try:
                backtest_path(year).unlink()
            except Exception:
                pass
            registry.pop(year, None)
            st.rerun()

        if run_clicked:
            # The backtest runs on the server in the background. Close the app, lock the phone,
            # come back later: the saved table keeps filling in race by race.
            rows = []
            if saved and saved.get("signature") == sig and not saved.get("complete"):
                rows = saved.get("rows", [])  # unfinished run with the same settings: resume it
            done_keys = {(r["Model"], r["Round"]) for r in rows}
            base_settings = {**BASELINE_SETTINGS, "fp": settings["fp"], "compounds": settings["compounds"]}
            models = [("This model", test_settings)] + ([("Previous model", base_settings)] if compare else [])
            queue = [(label_, int(ev["RoundNumber"]), ev, cfg)
                     for ev in done_rounds[:n_test] for label_, cfg in models
                     if (label_, int(ev["RoundNumber"])) not in done_keys]
            registry[year] = {"status": "running", "done": len(models) * n_test - len(queue),
                              "total": len(models) * n_test, "current": "starting...", "stop": False, "sig": sig}
            save_backtest(year, {"signature": sig, "rows": rows, "complete": not queue, "races": n_test,
                                 "updated": datetime.now().strftime("%d %b %Y %H:%M")})
            if queue:
                t = threading.Thread(target=_backtest_worker, args=(year, queue, sig, n_test, registry), daemon=True)
                try:  # lets the worker use Streamlit's data cache without warnings
                    from streamlit.runtime.scriptrunner import add_script_run_ctx
                    add_script_run_ctx(t)
                except Exception:
                    pass
                t.start()
            else:
                registry[year]["status"] = "done"
            st.rerun()

        if job and job.get("status") == "error":
            st.error(f"Backtest stopped with an error: {job.get('error')}")
        if running:
            total = max(job.get("total", 1), 1)
            st.progress(min(job.get("done", 0) / total, 1.0),
                        text=f"Backtest running in the background: {job.get('done', 0)} of {total} scored · "
                             f"now {job.get('current', '')}")
            st.caption("This page refreshes itself every few seconds. You can close it; the run continues "
                       "and the results below will be here when you come back.")
            saved = load_backtest(year)

        if saved and saved.get("rows"):
            bt = pd.DataFrame(saved["rows"]).sort_values(["Round", "Model"], ascending=[False, True])
            for col in ("Pole", "Winner"):  # JSON round-trip safety
                if col in bt.columns:
                    bt[col] = bt[col].map({True: True, False: False, "True": True, "False": False})
            for col in ("Q top3 hits", "Q avg error", "Q correlation", "Podium hits", "Podium exact",
                        "R avg error", "R correlation"):
                if col in bt.columns:
                    bt[col] = pd.to_numeric(bt[col], errors="coerce")
            st.caption(f"Saved results from {saved.get('updated', '?')} · settings: {saved.get('signature', '?')}")
            if saved.get("signature") != sig:
                st.info("Your Settings have changed since this backtest was run. Press Run backtest to re-score "
                        "with the current settings.")
            summary_rows = []
            for label_, g in bt.groupby("Model", sort=False):
                def pct(col):
                    return f"{g[col].mean() * 100:.0f}%" if col in g.columns and g[col].notna().any() else "–"

                def avg(col, nd=2):
                    return round(g[col].mean(), nd) if col in g.columns and g[col].notna().any() else "–"
                summary_rows.append({
                    "Model": label_,
                    "Races": int(g["Pole"].notna().sum()) if "Pole" in g.columns else 0,
                    "Pole right": pct("Pole"), "Quali top-3 hits (of 3)": avg("Q top3 hits"),
                    "Quali avg error (places)": avg("Q avg error", 1),
                    "Winner right": pct("Winner"), "Podium hits (of 3)": avg("Podium hits"),
                    "Podium exact (of 3)": avg("Podium exact"),
                    "Race avg error (places)": avg("R avg error", 1),
                    "Race correlation": avg("R correlation"),
                })
            st.markdown("**Summary**")
            show_table(pd.DataFrame(summary_rows))
            st.caption("Correlation runs from -1 to 1; 1 means the predicted order matched the result exactly. "
                       "Avg error is how many places each driver was off, on average.")
            st.markdown("**Race by race**")
            detail = bt.copy()
            for col in ("Pole", "Winner"):
                if col in detail.columns:
                    detail[col] = detail[col].map({True: "✓", False: "✗"})
            for col in ("Q avg error", "R avg error", "Q correlation", "R correlation"):
                if col in detail.columns:
                    detail[col] = pd.to_numeric(detail[col], errors="coerce").round(2)
            show_table(detail.drop(columns=["Round"]).fillna("–"))
            st.caption("If the previous model scores better on several races, lower the recent form weight "
                       "or turn off the team correction in Settings and run again.")
        if running:
            time.sleep(5)
            st.rerun()

st.caption("Built from FastF1 and OpenF1 data. Projections are estimates, not guarantees.")
