"""
Agent-modifiable training pipeline for March Madness predictions.
Modify this file to experiment with features, models, and hyperparameters.

Usage: python train.py
"""

import time
import math
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from prepare import (
    load_data, compute_season_stats, parse_seeds, compute_win_pct,
    _make_lookups, STAT_COLS, CLIP_LOW, CLIP_HIGH,
    VAL_SEASONS, evaluate,
)

# ── ELO Parameters (experiment with these) ───────────────────────────────────
INITIAL_ELO = 1500
K = 30
SEASON_REGRESSION = 1.00
HOME_COURT_BONUS = 0
TOURNEY_DAY_CUTOFF = 133
TOURNEY_WEIGHT = 3.0
MODEL_PATH = "model.joblib"


# ── Efficiency Stats ────────────────────────────────────────────────────────
EFFICIENCY_COLS = [
    "off_eff", "def_eff", "net_eff", "efg_pct", "opp_efg_pct",
    "to_rate", "opp_to_rate", "or_rate", "ft_rate", "tempo",
]


def compute_efficiency_stats(detailed_results):
    """Compute per-team per-season efficiency metrics from detailed results."""
    rows = []
    for _, g in detailed_results.iterrows():
        for is_winner in [True, False]:
            pf = "W" if is_winner else "L"
            op = "L" if is_winner else "W"
            score = int(g[f"{pf}Score"])
            opp_score = int(g[f"{op}Score"])
            fga = int(g[f"{pf}FGA"])
            fgm = int(g[f"{pf}FGM"])
            fga3 = int(g[f"{pf}FGA3"])
            fgm3 = int(g[f"{pf}FGM3"])
            fta = int(g[f"{pf}FTA"])
            orb = int(g[f"{pf}OR"])
            to = int(g[f"{pf}TO"])
            opp_fga = int(g[f"{op}FGA"])
            opp_fgm = int(g[f"{op}FGM"])
            opp_fga3 = int(g[f"{op}FGA3"])
            opp_fgm3 = int(g[f"{op}FGM3"])
            opp_fta = int(g[f"{op}FTA"])
            opp_orb = int(g[f"{op}OR"])
            opp_to = int(g[f"{op}TO"])

            # Possessions estimate (Kenpom-style)
            poss = fga - orb + to + 0.475 * fta
            opp_poss = opp_fga - opp_orb + opp_to + 0.475 * opp_fta
            avg_poss = (poss + opp_poss) / 2.0
            avg_poss = max(avg_poss, 1.0)

            off_eff = score / avg_poss * 100
            def_eff = opp_score / avg_poss * 100
            net_eff = off_eff - def_eff

            # Effective FG%
            efg_pct = (fgm + 0.5 * fgm3) / fga if fga > 0 else 0
            opp_efg_pct = (opp_fgm + 0.5 * opp_fgm3) / opp_fga if opp_fga > 0 else 0

            # Turnover rate, OR rate, FT rate
            to_rate = to / avg_poss if avg_poss > 0 else 0
            opp_to_rate = opp_to / avg_poss if avg_poss > 0 else 0
            drb = int(g[f"{op}DR"]) if f"{op}DR" in g else 0
            or_rate = orb / (orb + drb) if (orb + drb) > 0 else 0
            ft_rate = fta / fga if fga > 0 else 0

            rows.append({
                "Season": int(g["Season"]),
                "TeamID": int(g[f"{pf}TeamID"]),
                "off_eff": off_eff,
                "def_eff": def_eff,
                "net_eff": net_eff,
                "efg_pct": efg_pct,
                "opp_efg_pct": opp_efg_pct,
                "to_rate": to_rate,
                "opp_to_rate": opp_to_rate,
                "or_rate": or_rate,
                "ft_rate": ft_rate,
                "tempo": avg_poss,
            })

    df = pd.DataFrame(rows)
    agg = df.groupby(["Season", "TeamID"]).mean(numeric_only=True).reset_index()
    return agg


def _make_efficiency_lookup(eff_stats):
    """Build dict lookup from efficiency stats DataFrame."""
    lookup = {}
    for _, r in eff_stats.iterrows():
        lookup[(int(r["Season"]), int(r["TeamID"]))] = {
            c: r[c] for c in EFFICIENCY_COLS
        }
    return lookup


# ── Road/Neutral Record ────────────────────────────────────────────────────
def compute_road_record(compact_results):
    """Compute win% in away and neutral games per team per season."""
    records = {}  # (season, team) -> [wins, games]
    for _, g in compact_results.iterrows():
        s = int(g["Season"])
        w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
        loc = g["WLoc"]
        # Away wins for winner
        if loc == "A":  # winner played away
            records.setdefault((s, w_id), [0, 0])
            records[(s, w_id)][0] += 1
            records[(s, w_id)][1] += 1
            records.setdefault((s, l_id), [0, 0])
            records[(s, l_id)][1] += 1
        elif loc == "N":  # neutral site
            records.setdefault((s, w_id), [0, 0])
            records[(s, w_id)][0] += 1
            records[(s, w_id)][1] += 1
            records.setdefault((s, l_id), [0, 0])
            records[(s, l_id)][1] += 1
        # loc == "H" means winner at home — loser was away
        elif loc == "H":
            records.setdefault((s, l_id), [0, 0])
            records[(s, l_id)][1] += 1  # away loss
            # winner doesn't count (was at home)

    lookup = {}
    for k, (w, g) in records.items():
        lookup[k] = w / g if g > 0 else 0.5
    return lookup


# ── Close Game Performance ─────────────────────────────────────────────────
def compute_close_game_record(compact_results, margin=5):
    """Compute win% in games decided by ≤margin points."""
    records = {}  # (season, team) -> [wins, games]
    for _, g in compact_results.iterrows():
        s = int(g["Season"])
        w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
        diff = int(g["WScore"]) - int(g["LScore"])
        if diff <= margin:
            records.setdefault((s, w_id), [0, 0])
            records[(s, w_id)][0] += 1
            records[(s, w_id)][1] += 1
            records.setdefault((s, l_id), [0, 0])
            records[(s, l_id)][1] += 1
    lookup = {}
    for k, (w, g) in records.items():
        lookup[k] = w / g if g > 0 else 0.5
    return lookup


# ── Conference Tournament Results ──────────────────────────────────────────
def compute_conf_tourney_record():
    """Compute conference tournament win count per team per season."""
    try:
        df_m = pd.read_csv("MConferenceTourneyGames.csv")
        df_w = pd.read_csv("WConferenceTourneyGames.csv")
    except FileNotFoundError:
        return {}, {}

    m_lookup, w_lookup = {}, {}
    for df, lookup in [(df_m, m_lookup), (df_w, w_lookup)]:
        wins = df.groupby(["Season", "WTeamID"]).size().reset_index(name="conf_tourney_wins")
        wins.columns = ["Season", "TeamID", "conf_tourney_wins"]
        games_w = df.groupby(["Season", "WTeamID"]).size().reset_index(name="g")
        games_w.columns = ["Season", "TeamID", "g"]
        games_l = df.groupby(["Season", "LTeamID"]).size().reset_index(name="g")
        games_l.columns = ["Season", "TeamID", "g"]
        games = pd.concat([games_w, games_l]).groupby(["Season", "TeamID"])["g"].sum().reset_index()
        merged = wins.merge(games, on=["Season", "TeamID"], how="outer").fillna(0)
        merged["conf_tourney_wpct"] = merged["conf_tourney_wins"] / merged["g"].clip(lower=1)
        for _, r in merged.iterrows():
            lookup[(int(r["Season"]), int(r["TeamID"]))] = {
                "conf_tourney_wins": r["conf_tourney_wins"],
                "conf_tourney_wpct": r["conf_tourney_wpct"],
            }
    return m_lookup, w_lookup


# ── Power Conference ──────────────────────────────────────────────────────
POWER_CONFERENCES = {"acc", "big_east", "big_ten", "big_twelve", "pac_twelve", "sec"}


def compute_power_conf(prefix):
    """Build lookup: (season, team) -> 1 if power conference, else 0."""
    df = pd.read_csv(f"{prefix}TeamConferences.csv")
    lookup = {}
    for _, r in df.iterrows():
        lookup[(int(r["Season"]), int(r["TeamID"]))] = int(r["ConfAbbrev"] in POWER_CONFERENCES)
    return lookup


# ── Win Streak ────────────────────────────────────────────────────────────
def compute_win_streak(compact_results):
    """Compute end-of-regular-season win/loss streak per team per season.
    Positive = win streak, negative = loss streak."""
    sorted_games = compact_results.sort_values(["Season", "DayNum"]).reset_index(drop=True)
    streaks = {}  # (season, team) -> streak

    for _, g in sorted_games.iterrows():
        s = int(g["Season"])
        w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])

        # Winner: extend win streak or start new one
        prev_w = streaks.get((s, w_id), 0)
        streaks[(s, w_id)] = (prev_w + 1) if prev_w >= 0 else 1

        # Loser: extend loss streak or start new one
        prev_l = streaks.get((s, l_id), 0)
        streaks[(s, l_id)] = (prev_l - 1) if prev_l <= 0 else -1

    return streaks


# ── Last N Games Stats ─────────────────────────────────────────────────────
def compute_last_n_stats(detailed_results, n=10):
    """Compute stats from only the last N games of each season per team."""
    # Sort by season and day
    df = detailed_results.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    rows = []
    for _, g in df.iterrows():
        for is_winner in [True, False]:
            pf = "W" if is_winner else "L"
            op = "L" if is_winner else "W"
            rows.append({
                "Season": int(g["Season"]),
                "TeamID": int(g[f"{pf}TeamID"]),
                "DayNum": int(g["DayNum"]),
                "score": int(g[f"{pf}Score"]),
                "opp_score": int(g[f"{op}Score"]),
                "fg_pct": int(g[f"{pf}FGM"]) / max(int(g[f"{pf}FGA"]), 1),
                "to": int(g[f"{pf}TO"]),
            })

    game_df = pd.DataFrame(rows)
    game_df = game_df.sort_values(["Season", "TeamID", "DayNum"])

    # Take last N games per team per season
    last_n = game_df.groupby(["Season", "TeamID"]).tail(n)
    agg = last_n.groupby(["Season", "TeamID"]).agg({
        "score": "mean",
        "opp_score": "mean",
        "fg_pct": "mean",
        "to": "mean",
    }).reset_index()
    agg.columns = ["Season", "TeamID", "last_n_ppg", "last_n_opp_ppg", "last_n_fg_pct", "last_n_to"]
    agg["last_n_margin"] = agg["last_n_ppg"] - agg["last_n_opp_ppg"]

    lookup = {}
    for _, r in agg.iterrows():
        lookup[(int(r["Season"]), int(r["TeamID"]))] = {
            "last_n_ppg": r["last_n_ppg"],
            "last_n_margin": r["last_n_margin"],
            "last_n_fg_pct": r["last_n_fg_pct"],
            "last_n_to": r["last_n_to"],
        }
    return lookup

LAST_N_COLS = ["last_n_ppg", "last_n_margin", "last_n_fg_pct", "last_n_to"]


# ── Massey Ordinals ─────────────────────────────────────────────────────────
ELITE_SYSTEMS = ["POM", "SAG", "MOR", "COL", "BPI", "RPI", "MAS", "WIL", "DOK", "KPK", "DII", "PGH", "TRK", "TRP", "INC", "HAS", "BWE", "EMK", "JNG", "DUN"]


def compute_massey_ranks():
    """Load Massey Ordinals and compute rank features from all + elite systems, plus trajectory."""
    print("  Loading Massey Ordinals...")
    raw = pd.read_csv("MMasseyOrdinals.csv")

    # End-of-season ranks (max day)
    max_days = raw.groupby("Season")["RankingDayNum"].max().reset_index()
    max_days.columns = ["Season", "MaxDay"]
    df = raw.merge(max_days, on="Season")
    df = df[df["RankingDayNum"] == df["MaxDay"]]

    # Average rank across ALL systems per team per season
    avg_ranks = df.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index()
    avg_ranks.columns = ["Season", "TeamID", "massey_avg_rank"]
    # Median and best rank across all systems
    agg = df.groupby(["Season", "TeamID"])["OrdinalRank"].agg(["median", "min"]).reset_index()
    agg.columns = ["Season", "TeamID", "massey_median_rank", "massey_best_rank"]
    avg_ranks = avg_ranks.merge(agg, on=["Season", "TeamID"])

    # Elite systems average
    elite_df = df[df["SystemName"].isin(ELITE_SYSTEMS)]
    if len(elite_df) > 0:
        elite_avg = elite_df.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index()
        elite_avg.columns = ["Season", "TeamID", "massey_elite_rank"]
        avg_ranks = avg_ranks.merge(elite_avg, on=["Season", "TeamID"], how="left")
    else:
        avg_ranks["massey_elite_rank"] = np.nan

    # Trajectory: mid-season avg rank vs end-of-season avg rank
    # Mid-season = around day 70 (midpoint of ~133 day season)
    mid_df = raw[(raw["RankingDayNum"] >= 60) & (raw["RankingDayNum"] <= 80)]
    if len(mid_df) > 0:
        mid_avg = mid_df.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index()
        mid_avg.columns = ["Season", "TeamID", "massey_mid_rank"]
        avg_ranks = avg_ranks.merge(mid_avg, on=["Season", "TeamID"], how="left")
        avg_ranks["massey_trajectory"] = avg_ranks["massey_mid_rank"] - avg_ranks["massey_avg_rank"]
    else:
        avg_ranks["massey_mid_rank"] = np.nan
        avg_ranks["massey_trajectory"] = np.nan

    # Build lookup
    massey_lookup = {}
    for _, r in avg_ranks.iterrows():
        massey_lookup[(int(r["Season"]), int(r["TeamID"]))] = {
            "massey_avg_rank": r["massey_avg_rank"],
            "massey_median_rank": r["massey_median_rank"],
            "massey_best_rank": r["massey_best_rank"],
            "massey_elite_rank": r["massey_elite_rank"],
            "massey_trajectory": r["massey_trajectory"],
        }
    return massey_lookup

# ── XGBoost Hyperparameters (experiment with these) ──────────────────────────
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "nthread": -1,
    "verbosity": 0,
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha": 0.3,
    "reg_lambda": 2.0,
    "gamma": 0.0,
}


# ── ELO Ratings ──────────────────────────────────────────────────────────────
def compute_elo_ratings(compact_results, tourney_results):
    """
    Compute ELO ratings from compact results.
    Returns a dict {(season, team_id): elo} snapshotted BEFORE tournament play.
    Tournament games still update ELO for carry-forward into next season.
    """
    elo = {}

    reg = compact_results.copy()
    reg["is_tourney"] = 0
    trn = tourney_results.copy()
    trn["is_tourney"] = 1
    all_games = pd.concat([reg, trn], ignore_index=True)
    all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    seasons = sorted(all_games["Season"].unique())
    elo_snapshot = {}

    for season in seasons:
        for tid in elo:
            elo[tid] = INITIAL_ELO + SEASON_REGRESSION * (elo[tid] - INITIAL_ELO)

        season_games = all_games[all_games["Season"] == season]
        reg_games = season_games[season_games["DayNum"] <= TOURNEY_DAY_CUTOFF]
        tourney_games = season_games[season_games["DayNum"] > TOURNEY_DAY_CUTOFF]

        for _, row in reg_games.iterrows():
            _update_elo(elo, row)

        teams_this_season = set(season_games["WTeamID"]) | set(season_games["LTeamID"])
        for tid in teams_this_season:
            elo_snapshot[(season, tid)] = elo.get(tid, INITIAL_ELO)

        for _, row in tourney_games.iterrows():
            _update_elo(elo, row)

    return elo_snapshot


def _update_elo(elo, row):
    w_id, l_id = int(row["WTeamID"]), int(row["LTeamID"])
    w_elo = elo.get(w_id, INITIAL_ELO)
    l_elo = elo.get(l_id, INITIAL_ELO)

    w_adj, l_adj = 0, 0
    if row["WLoc"] == "H":
        w_adj = HOME_COURT_BONUS
    elif row["WLoc"] == "A":
        l_adj = HOME_COURT_BONUS

    expected_w = 1.0 / (1.0 + 10.0 ** ((l_elo + l_adj - w_elo - w_adj) / 400.0))

    margin = abs(int(row["WScore"]) - int(row["LScore"]))
    mov_mult = min(math.sqrt(margin), 2.5)

    update = K * mov_mult * (1 - expected_w)
    elo[w_id] = w_elo + update
    elo[l_id] = l_elo - update


# ── Feature Engineering ──────────────────────────────────────────────────────
def _build_feature_row(season, t1, t2, elo_snapshot, seeds_lookup, stats_lookup, winpct_lookup, is_mens, massey_lookup=None, eff_lookup=None, road_lookup=None, close_lookup=None, conf_tourney_lookup=None, streak_lookup=None, power_lookup=None, last_n_lookup=None):
    """Build a single feature vector for a T1 vs T2 matchup."""
    row = {"Season": season, "T1": t1, "T2": t2, "is_mens": int(is_mens)}

    # ELO
    row["T1_elo"] = elo_snapshot.get((season, t1), INITIAL_ELO)
    row["T2_elo"] = elo_snapshot.get((season, t2), INITIAL_ELO)
    row["elo_diff"] = row["T1_elo"] - row["T2_elo"]

    # Seeds
    row["T1_seed"] = seeds_lookup.get((season, t1), np.nan)
    row["T2_seed"] = seeds_lookup.get((season, t2), np.nan)
    row["seed_diff"] = row["T1_seed"] - row["T2_seed"] if not (
        np.isnan(row["T1_seed"]) if isinstance(row["T1_seed"], float) else False
    ) and not (
        np.isnan(row["T2_seed"]) if isinstance(row["T2_seed"], float) else False
    ) else np.nan

    # Win %
    row["T1_win_pct"] = winpct_lookup.get((season, t1), np.nan)
    row["T2_win_pct"] = winpct_lookup.get((season, t2), np.nan)
    row["win_pct_diff"] = (
        row["T1_win_pct"] - row["T2_win_pct"]
        if not pd.isna(row["T1_win_pct"]) and not pd.isna(row["T2_win_pct"])
        else np.nan
    )

    # Box-score stats
    t1_stats = stats_lookup.get((season, t1), {})
    t2_stats = stats_lookup.get((season, t2), {})
    for col in STAT_COLS:
        v1 = t1_stats.get(col, np.nan)
        v2 = t2_stats.get(col, np.nan)
        row[f"T1_{col}"] = v1
        row[f"T2_{col}"] = v2
        if pd.notna(v1) and pd.notna(v2):
            row[f"{col}_diff"] = v1 - v2
        else:
            row[f"{col}_diff"] = np.nan

    # Massey Ordinals (Men's only)
    if massey_lookup is not None and is_mens:
        t1_massey = massey_lookup.get((season, t1), {})
        t2_massey = massey_lookup.get((season, t2), {})
        for col in ["massey_avg_rank", "massey_median_rank", "massey_best_rank", "massey_elite_rank", "massey_trajectory"]:
            v1 = t1_massey.get(col, np.nan)
            v2 = t2_massey.get(col, np.nan)
            row[f"T1_{col}"] = v1
            row[f"T2_{col}"] = v2
            if pd.notna(v1) and pd.notna(v2):
                row[f"{col}_diff"] = v1 - v2
            else:
                row[f"{col}_diff"] = np.nan
    elif massey_lookup is not None:
        for col in ["massey_avg_rank", "massey_median_rank", "massey_best_rank", "massey_elite_rank", "massey_trajectory"]:
            row[f"T1_{col}"] = np.nan
            row[f"T2_{col}"] = np.nan
            row[f"{col}_diff"] = np.nan

    # Efficiency stats
    if eff_lookup is not None:
        t1_eff = eff_lookup.get((season, t1), {})
        t2_eff = eff_lookup.get((season, t2), {})
        for col in EFFICIENCY_COLS:
            v1 = t1_eff.get(col, np.nan)
            v2 = t2_eff.get(col, np.nan)
            row[f"T1_{col}"] = v1
            row[f"T2_{col}"] = v2
            if pd.notna(v1) and pd.notna(v2):
                row[f"{col}_diff"] = v1 - v2
            else:
                row[f"{col}_diff"] = np.nan

    # Road/neutral record
    if road_lookup is not None:
        row["T1_road_pct"] = road_lookup.get((season, t1), 0.5)
        row["T2_road_pct"] = road_lookup.get((season, t2), 0.5)
        row["road_pct_diff"] = row["T1_road_pct"] - row["T2_road_pct"]

    # Close game performance
    if close_lookup is not None:
        row["T1_close_pct"] = close_lookup.get((season, t1), 0.5)
        row["T2_close_pct"] = close_lookup.get((season, t2), 0.5)
        row["close_pct_diff"] = row["T1_close_pct"] - row["T2_close_pct"]

    # Win streak
    if streak_lookup is not None:
        row["T1_streak"] = streak_lookup.get((season, t1), 0)
        row["T2_streak"] = streak_lookup.get((season, t2), 0)
        row["streak_diff"] = row["T1_streak"] - row["T2_streak"]

    # Conference tournament results
    if conf_tourney_lookup is not None:
        t1_ct = conf_tourney_lookup.get((season, t1), {})
        t2_ct = conf_tourney_lookup.get((season, t2), {})
        for col in ["conf_tourney_wins", "conf_tourney_wpct"]:
            row[f"T1_{col}"] = t1_ct.get(col, 0)
            row[f"T2_{col}"] = t2_ct.get(col, 0)
            row[f"{col}_diff"] = row[f"T1_{col}"] - row[f"T2_{col}"]

    # Last N games stats
    if last_n_lookup is not None:
        t1_ln = last_n_lookup.get((season, t1), {})
        t2_ln = last_n_lookup.get((season, t2), {})
        for col in LAST_N_COLS:
            v1 = t1_ln.get(col, np.nan)
            v2 = t2_ln.get(col, np.nan)
            row[f"T1_{col}"] = v1
            row[f"T2_{col}"] = v2
            if pd.notna(v1) and pd.notna(v2):
                row[f"{col}_diff"] = v1 - v2
            else:
                row[f"{col}_diff"] = np.nan

    # Power conference indicator
    if power_lookup is not None:
        row["T1_power_conf"] = power_lookup.get((season, t1), 0)
        row["T2_power_conf"] = power_lookup.get((season, t2), 0)
        row["power_conf_diff"] = row["T1_power_conf"] - row["T2_power_conf"]

    return row


# ── Training Data Construction ───────────────────────────────────────────────
def build_training_data(data, elo_m, elo_w, stats_m, stats_w, seeds_m, seeds_w, wp_m, wp_w, massey_lookup=None, eff_m=None, eff_w=None, road_m=None, road_w=None, close_m=None, close_w=None, conf_m=None, conf_w=None, streak_m=None, streak_w=None, power_m=None, power_w=None, last_n_m=None, last_n_w=None):
    """Build training matrix from regular season + tournament games for both genders."""
    seeds_look_m, stats_look_m, wp_look_m = _make_lookups(seeds_m, stats_m, wp_m)
    seeds_look_w, stats_look_w, wp_look_w = _make_lookups(seeds_w, stats_w, wp_w)
    eff_look_m = _make_efficiency_lookup(eff_m) if eff_m is not None else None
    eff_look_w = _make_efficiency_lookup(eff_w) if eff_w is not None else None

    rows = []
    for gender, label in [("mens", True), ("womens", False)]:
        elo_snap = elo_m if label else elo_w
        seeds_look = seeds_look_m if label else seeds_look_w
        stats_look = stats_look_m if label else stats_look_w
        wp_look = wp_look_m if label else wp_look_w
        eff_look = eff_look_m if label else eff_look_w
        road_look = road_m if label else road_w
        close_look = close_m if label else close_w
        conf_look = conf_m if label else conf_w
        streak_look = streak_m if label else streak_w
        power_look = power_m if label else power_w
        last_n_look = last_n_m if label else last_n_w

        compact = data[f"{gender}_compact"]
        for _, g in compact.iterrows():
            s = int(g["Season"])
            w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
            t1, t2 = min(w_id, l_id), max(w_id, l_id)
            target = 1.0 if t1 == w_id else 0.0

            row = _build_feature_row(s, t1, t2, elo_snap, seeds_look, stats_look, wp_look, label, massey_lookup, eff_look, road_look, close_look, conf_look, streak_look, power_look, last_n_look)
            row["target"] = target
            row["weight"] = 1.0
            rows.append(row)

        tourney = data[f"{gender}_tourney"]
        for _, g in tourney.iterrows():
            s = int(g["Season"])
            w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
            t1, t2 = min(w_id, l_id), max(w_id, l_id)
            target = 1.0 if t1 == w_id else 0.0

            row = _build_feature_row(s, t1, t2, elo_snap, seeds_look, stats_look, wp_look, label, massey_lookup, eff_look, road_look, close_look, conf_look, streak_look, power_look, last_n_look)
            row["target"] = target
            row["weight"] = TOURNEY_WEIGHT
            row["is_tourney"] = 1
            rows.append(row)

    df = pd.DataFrame(rows)
    if "is_tourney" not in df.columns:
        df["is_tourney"] = 0
    df["is_tourney"] = df["is_tourney"].fillna(0).astype(int)
    # Recency weighting: more recent seasons get higher weight
    max_season = df["Season"].max()
    df["weight"] = df["weight"] * (1.0 + 0.05 * (df["Season"] - 2003))
    return df


def get_feature_cols(df):
    """Get feature column names (everything except metadata/target)."""
    exclude = {"Season", "T1", "T2", "target", "weight", "is_tourney"}
    return [c for c in df.columns if c not in exclude]


# ── Model Training ───────────────────────────────────────────────────────────
def train_final_model(train_df, feature_cols, params=None):
    """Train a single model on all provided data. Used by inference.py."""
    if params is None:
        params = XGB_PARAMS.copy()
    else:
        params = {**XGB_PARAMS, **params}
        # Ensure fixed params are set
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        params["tree_method"] = "hist"
        params["nthread"] = -1
        params["verbosity"] = 0
    model = xgb.XGBClassifier(**params)
    model.fit(
        train_df[feature_cols],
        train_df["target"],
        sample_weight=train_df["weight"],
        verbose=False,
    )
    return model


def train_and_predict(data):
    """
    Run expanding-window CV on tournament years.
    Returns {season: (y_true, y_pred)}.
    """
    # Compute all derived features
    print("  Computing ELO ratings...")
    elo_m = compute_elo_ratings(data["mens_compact"], data["mens_tourney"])
    elo_w = compute_elo_ratings(data["womens_compact"], data["womens_tourney"])

    print("  Computing season stats...")
    stats_m = compute_season_stats(data["mens_detailed"])
    stats_w = compute_season_stats(data["womens_detailed"])

    print("  Parsing seeds and win pct...")
    seeds_m = parse_seeds(data["mens_seeds"])
    seeds_w = parse_seeds(data["womens_seeds"])
    wp_m = compute_win_pct(data["mens_compact"])
    wp_w = compute_win_pct(data["womens_compact"])

    massey_lookup = compute_massey_ranks()

    print("  Computing efficiency stats...")
    eff_m = compute_efficiency_stats(data["mens_detailed"])
    eff_w = compute_efficiency_stats(data["womens_detailed"])

    print("  Computing road/neutral records...")
    road_m = compute_road_record(data["mens_compact"])
    road_w = compute_road_record(data["womens_compact"])

    print("  Computing close game records...")
    close_m = compute_close_game_record(data["mens_compact"])
    close_w = compute_close_game_record(data["womens_compact"])

    print("  Computing conference tournament results...")
    conf_m, conf_w = compute_conf_tourney_record()

    print("  Computing win streaks...")
    streak_m = compute_win_streak(data["mens_compact"])
    streak_w = compute_win_streak(data["womens_compact"])

    print("  Computing power conference indicators...")
    power_m = compute_power_conf("M")
    power_w = compute_power_conf("W")

    print("  Computing last-N game stats...")
    last_n_m = compute_last_n_stats(data["mens_detailed"])
    last_n_w = compute_last_n_stats(data["womens_detailed"])

    print("  Building training data...")
    train_df = build_training_data(
        data, elo_m, elo_w, stats_m, stats_w, seeds_m, seeds_w, wp_m, wp_w, massey_lookup, eff_m, eff_w, road_m, road_w, close_m, close_w, conf_m, conf_w, streak_m, streak_w, power_m, power_w, last_n_m, last_n_w
    )
    feature_cols = get_feature_cols(train_df)
    print(f"  Samples: {len(train_df):,}  Features: {len(feature_cols)}")

    # Feature selection will be done per fold
    n_keep = max(int(len(feature_cols) * 0.55), 5)
    print(f"  Will keep top {n_keep}/{len(feature_cols)} features per fold")

    # Build seed-matchup win rate prior from historical tournament data
    print("  Computing seed-matchup priors...")
    seed_prior = {}  # (seed1, seed2) -> historical win rate for seed1
    all_hist_tourney = pd.concat([data["mens_tourney"], data["womens_tourney"]], ignore_index=True)
    seeds_look_m, _, _ = _make_lookups(seeds_m, stats_m, wp_m)
    seeds_look_w_tmp, _, _ = _make_lookups(seeds_w, stats_w, wp_w)
    all_seeds = {**seeds_look_m, **seeds_look_w_tmp}

    seed_records = {}  # (s1, s2) -> [wins_for_s1, total]
    for _, g in all_hist_tourney.iterrows():
        s = int(g["Season"])
        w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
        t1, t2 = min(w_id, l_id), max(w_id, l_id)
        s1 = all_seeds.get((s, t1), None)
        s2 = all_seeds.get((s, t2), None)
        if s1 is not None and s2 is not None:
            key = (int(s1), int(s2))
            seed_records.setdefault(key, [0, 0])
            seed_records[key][1] += 1
            if t1 == w_id:
                seed_records[key][0] += 1

    for k, (w, t) in seed_records.items():
        seed_prior[k] = w / t if t > 0 else 0.5

    BLEND_ALPHA = 0.60  # 60% prior blend

    # Expanding-window CV — evaluate on tournament + regular season
    results = {}
    for val_season in VAL_SEASONS:
        tr = train_df[train_df["Season"] < val_season]
        va_tourney = train_df[(train_df["Season"] == val_season) & (train_df["is_tourney"] == 1)]
        va_reg = train_df[(train_df["Season"] == val_season) & (train_df["is_tourney"] == 0)]
        va = pd.concat([va_tourney, va_reg], ignore_index=True)

        if len(va) == 0:
            print(f"  {val_season}: no games, skipping")
            continue

        # Per-fold feature selection
        sel_model = xgb.XGBClassifier(**XGB_PARAMS)
        sel_model.fit(tr[feature_cols], tr["target"], sample_weight=tr["weight"], verbose=False)
        importances = sel_model.feature_importances_
        imp_order = np.argsort(importances)[::-1]
        selected_cols = [feature_cols[i] for i in imp_order[:n_keep]]

        # XGB model
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(
            tr[selected_cols], tr["target"],
            sample_weight=tr["weight"],
            verbose=False,
        )
        xgb_preds = xgb_model.predict_proba(va[selected_cols])[:, 1]

        # LightGBM model
        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            max_depth=5,
            learning_rate=0.03,
            n_estimators=800,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.3,
            reg_lambda=2.0,
            num_leaves=31,
            verbose=-1,
        )
        lgb_model.fit(
            tr[selected_cols], tr["target"],
            sample_weight=tr["weight"],
        )
        lgb_preds = lgb_model.predict_proba(va[selected_cols])[:, 1]

        # Ensemble: 70% XGB + 30% LightGBM
        preds = 0.7 * xgb_preds + 0.3 * lgb_preds

        # Blend with seed prior for tournament games
        for i in range(len(va)):
            if va.iloc[i].get("is_tourney", 0) == 1:
                s1 = va.iloc[i].get("T1_seed", None)
                s2 = va.iloc[i].get("T2_seed", None)
                if pd.notna(s1) and pd.notna(s2):
                    prior = seed_prior.get((int(s1), int(s2)), 0.5)
                    preds[i] = (1 - BLEND_ALPHA) * preds[i] + BLEND_ALPHA * prior

        results[val_season] = (va["target"].values, preds)
        print(f"  {val_season}: {len(va)} games ({len(va_tourney)} tourney + {len(va_reg)} reg season)")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    print("Loading data...")
    data = load_data()

    print("Training and evaluating...")
    results = train_and_predict(data)

    t1 = time.time()
    total_seconds = t1 - t0

    mean_ll, fold_losses = evaluate(results)

    per_fold = " ".join(f"{s}={ll:.4f}" for s, ll in sorted(fold_losses.items()))

    print("---")
    print(f"val_logloss:      {mean_ll:.6f}")
    print(f"per_fold:         {per_fold}")
    print(f"total_seconds:    {total_seconds:.1f}")
