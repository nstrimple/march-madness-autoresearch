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

from prepare import (
    load_data, compute_season_stats, parse_seeds, compute_win_pct,
    _make_lookups, STAT_COLS, CLIP_LOW, CLIP_HIGH,
    VAL_SEASONS, evaluate,
)

# ── ELO Parameters (experiment with these) ───────────────────────────────────
INITIAL_ELO = 1500
K = 20
SEASON_REGRESSION = 0.75
HOME_COURT_BONUS = 100
TOURNEY_DAY_CUTOFF = 133
TOURNEY_WEIGHT = 4.0
MODEL_PATH = "model.joblib"

# ── XGBoost Hyperparameters (experiment with these) ──────────────────────────
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "nthread": -1,
    "verbosity": 0,
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.1,
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
    mov_mult = min(math.log(margin + 1), 2.5)

    update = K * mov_mult * (1 - expected_w)
    elo[w_id] = w_elo + update
    elo[l_id] = l_elo - update


# ── Feature Engineering ──────────────────────────────────────────────────────
def _build_feature_row(season, t1, t2, elo_snapshot, seeds_lookup, stats_lookup, winpct_lookup, is_mens):
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

    return row


# ── Training Data Construction ───────────────────────────────────────────────
def build_training_data(data, elo_m, elo_w, stats_m, stats_w, seeds_m, seeds_w, wp_m, wp_w):
    """Build training matrix from regular season + tournament games for both genders."""
    seeds_look_m, stats_look_m, wp_look_m = _make_lookups(seeds_m, stats_m, wp_m)
    seeds_look_w, stats_look_w, wp_look_w = _make_lookups(seeds_w, stats_w, wp_w)

    rows = []
    for gender, label in [("mens", True), ("womens", False)]:
        elo_snap = elo_m if label else elo_w
        seeds_look = seeds_look_m if label else seeds_look_w
        stats_look = stats_look_m if label else stats_look_w
        wp_look = wp_look_m if label else wp_look_w

        compact = data[f"{gender}_compact"]
        for _, g in compact.iterrows():
            s = int(g["Season"])
            w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
            t1, t2 = min(w_id, l_id), max(w_id, l_id)
            target = 1.0 if t1 == w_id else 0.0

            row = _build_feature_row(s, t1, t2, elo_snap, seeds_look, stats_look, wp_look, label)
            row["target"] = target
            row["weight"] = 1.0
            rows.append(row)

        tourney = data[f"{gender}_tourney"]
        for _, g in tourney.iterrows():
            s = int(g["Season"])
            w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
            t1, t2 = min(w_id, l_id), max(w_id, l_id)
            target = 1.0 if t1 == w_id else 0.0

            row = _build_feature_row(s, t1, t2, elo_snap, seeds_look, stats_look, wp_look, label)
            row["target"] = target
            row["weight"] = TOURNEY_WEIGHT
            row["is_tourney"] = 1
            rows.append(row)

    df = pd.DataFrame(rows)
    if "is_tourney" not in df.columns:
        df["is_tourney"] = 0
    df["is_tourney"] = df["is_tourney"].fillna(0).astype(int)
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

    print("  Building training data...")
    train_df = build_training_data(
        data, elo_m, elo_w, stats_m, stats_w, seeds_m, seeds_w, wp_m, wp_w
    )
    feature_cols = get_feature_cols(train_df)
    print(f"  Samples: {len(train_df):,}  Features: {len(feature_cols)}")

    # Expanding-window CV
    results = {}
    for val_season in VAL_SEASONS:
        tr = train_df[train_df["Season"] < val_season]
        va = train_df[(train_df["Season"] == val_season) & (train_df["is_tourney"] == 1)]

        if len(va) == 0:
            print(f"  {val_season}: no tournament games, skipping")
            continue

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            tr[feature_cols], tr["target"],
            sample_weight=tr["weight"],
            verbose=False,
        )

        preds = model.predict_proba(va[feature_cols])[:, 1]
        results[val_season] = (va["target"].values, preds)
        print(f"  {val_season}: {len(va)} games evaluated")

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
