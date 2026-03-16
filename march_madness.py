"""
March Machine Learning Mania 2026 — XGBoost + ELO prediction pipeline.
Trains a unified men's + women's model with Optuna hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import math
import joblib
from sklearn.metrics import log_loss

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = ""
INITIAL_ELO = 1500
K = 20
SEASON_REGRESSION = 0.75
HOME_COURT_BONUS = 100
CLIP_LOW, CLIP_HIGH = 0.05, 0.95
MODEL_PATH = "model.joblib"
TOURNEY_WEIGHT = 4.0
TOURNEY_DAY_CUTOFF = 133  # regular season ends at DayNum 133


# ── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data():
    data = {}
    for prefix, label in [("M", "mens"), ("W", "womens")]:
        data[f"{label}_compact"] = pd.read_csv(f"{DATA_DIR}{prefix}RegularSeasonCompactResults.csv")
        data[f"{label}_detailed"] = pd.read_csv(f"{DATA_DIR}{prefix}RegularSeasonDetailedResults.csv")
        data[f"{label}_tourney"] = pd.read_csv(f"{DATA_DIR}{prefix}NCAATourneyCompactResults.csv")
        data[f"{label}_seeds"] = pd.read_csv(f"{DATA_DIR}{prefix}NCAATourneySeeds.csv")
    data["submission"] = pd.read_csv(f"{DATA_DIR}SampleSubmissionStage1.csv")
    return data


# ── 2. ELO Ratings ───────────────────────────────────────────────────────────
def compute_elo_ratings(compact_results, tourney_results):
    """
    Compute ELO ratings from compact results.
    Returns a dict {(season, team_id): elo} snapshotted BEFORE tournament play.
    Tournament games still update ELO for carry-forward into next season.
    """
    elo = {}  # team_id -> current elo

    # Combine and sort all games
    reg = compact_results.copy()
    reg["is_tourney"] = 0
    trn = tourney_results.copy()
    trn["is_tourney"] = 1
    all_games = pd.concat([reg, trn], ignore_index=True)
    all_games = all_games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    seasons = sorted(all_games["Season"].unique())
    elo_snapshot = {}  # (season, team_id) -> pre-tournament elo

    for season in seasons:
        # Season regression
        for tid in elo:
            elo[tid] = INITIAL_ELO + SEASON_REGRESSION * (elo[tid] - INITIAL_ELO)

        season_games = all_games[all_games["Season"] == season]
        reg_games = season_games[season_games["DayNum"] <= TOURNEY_DAY_CUTOFF]
        tourney_games = season_games[season_games["DayNum"] > TOURNEY_DAY_CUTOFF]

        # Process regular season
        for _, row in reg_games.iterrows():
            _update_elo(elo, row)

        # Snapshot after regular season, before tournament
        teams_this_season = set(season_games["WTeamID"]) | set(season_games["LTeamID"])
        for tid in teams_this_season:
            elo_snapshot[(season, tid)] = elo.get(tid, INITIAL_ELO)

        # Process tournament (updates carry into next season)
        for _, row in tourney_games.iterrows():
            _update_elo(elo, row)

    return elo_snapshot


def _update_elo(elo, row):
    w_id, l_id = int(row["WTeamID"]), int(row["LTeamID"])
    w_elo = elo.get(w_id, INITIAL_ELO)
    l_elo = elo.get(l_id, INITIAL_ELO)

    # Home court adjustment for expected score calculation
    w_adj, l_adj = 0, 0
    if row["WLoc"] == "H":
        w_adj = HOME_COURT_BONUS
    elif row["WLoc"] == "A":
        l_adj = HOME_COURT_BONUS

    expected_w = 1.0 / (1.0 + 10.0 ** ((l_elo + l_adj - w_elo - w_adj) / 400.0))

    # Margin of victory multiplier
    margin = abs(int(row["WScore"]) - int(row["LScore"]))
    mov_mult = min(math.log(margin + 1), 2.5)

    update = K * mov_mult * (1 - expected_w)
    elo[w_id] = w_elo + update
    elo[l_id] = l_elo - update


# ── 3. Season Stats ──────────────────────────────────────────────────────────
def compute_season_stats(detailed_results):
    """Compute per-team per-season average box-score stats from detailed results."""
    rows = []
    for _, g in detailed_results.iterrows():
        # Winner perspective
        rows.append(_team_game_stats(g, is_winner=True))
        # Loser perspective
        rows.append(_team_game_stats(g, is_winner=False))

    df = pd.DataFrame(rows)
    agg = df.groupby(["Season", "TeamID"]).mean(numeric_only=True).reset_index()
    return agg


def _team_game_stats(g, is_winner):
    pf = "W" if is_winner else "L"  # prefix for this team
    op = "L" if is_winner else "W"  # opponent prefix
    score = int(g[f"{pf}Score"])
    opp_score = int(g[f"{op}Score"])

    fga = int(g[f"{pf}FGA"])
    fgm = int(g[f"{pf}FGM"])
    fga3 = int(g[f"{pf}FGA3"])
    fgm3 = int(g[f"{pf}FGM3"])
    fta = int(g[f"{pf}FTA"])
    ftm = int(g[f"{pf}FTM"])

    opp_fga = int(g[f"{op}FGA"])
    opp_fgm = int(g[f"{op}FGM"])
    opp_fga3 = int(g[f"{op}FGA3"])
    opp_fgm3 = int(g[f"{op}FGM3"])

    ast = int(g[f"{pf}Ast"])
    to = int(g[f"{pf}TO"])
    opp_to = int(g[f"{op}TO"])

    return {
        "Season": int(g["Season"]),
        "TeamID": int(g[f"{pf}TeamID"]),
        "fg_pct": fgm / fga if fga > 0 else 0,
        "fg3_pct": fgm3 / fga3 if fga3 > 0 else 0,
        "ft_pct": ftm / fta if fta > 0 else 0,
        "ppg": score,
        "opp_ppg": opp_score,
        "score_diff": score - opp_score,
        "or_pg": int(g[f"{pf}OR"]),
        "dr_pg": int(g[f"{pf}DR"]),
        "ast_pg": ast,
        "to_pg": to,
        "ast_to_ratio": ast / to if to > 0 else ast,
        "stl_pg": int(g[f"{pf}Stl"]),
        "blk_pg": int(g[f"{pf}Blk"]),
        "opp_fg_pct": opp_fgm / opp_fga if opp_fga > 0 else 0,
        "opp_fg3_pct": opp_fgm3 / opp_fga3 if opp_fga3 > 0 else 0,
        "opp_to_pg": opp_to,
    }


# ── 4. Seeds & Win % ─────────────────────────────────────────────────────────
def parse_seeds(seeds_df):
    """Extract numeric seed from seed string (e.g., 'W01' -> 1, 'Z16a' -> 16)."""
    seeds_df = seeds_df.copy()
    seeds_df["SeedNum"] = seeds_df["Seed"].apply(
        lambda s: int("".join(filter(str.isdigit, s)))
    )
    return seeds_df[["Season", "TeamID", "SeedNum"]]


def compute_win_pct(compact_results):
    """Compute regular season win percentage per team per season."""
    wins = compact_results.groupby(["Season", "WTeamID"]).size().reset_index(name="wins")
    wins.columns = ["Season", "TeamID", "wins"]
    losses = compact_results.groupby(["Season", "LTeamID"]).size().reset_index(name="losses")
    losses.columns = ["Season", "TeamID", "losses"]

    wl = pd.merge(wins, losses, on=["Season", "TeamID"], how="outer").fillna(0)
    wl["win_pct"] = wl["wins"] / (wl["wins"] + wl["losses"])
    return wl[["Season", "TeamID", "win_pct"]]


# ── 5. Build Feature Matrix ──────────────────────────────────────────────────
STAT_COLS = [
    "fg_pct", "fg3_pct", "ft_pct", "ppg", "opp_ppg", "score_diff",
    "or_pg", "dr_pg", "ast_pg", "to_pg", "ast_to_ratio",
    "stl_pg", "blk_pg", "opp_fg_pct", "opp_fg3_pct", "opp_to_pg",
]


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


def _make_lookups(seeds_parsed, season_stats, win_pcts):
    seeds_lookup = {}
    for _, r in seeds_parsed.iterrows():
        seeds_lookup[(int(r["Season"]), int(r["TeamID"]))] = r["SeedNum"]

    stats_lookup = {}
    for _, r in season_stats.iterrows():
        stats_lookup[(int(r["Season"]), int(r["TeamID"]))] = {
            c: r[c] for c in STAT_COLS if c in r
        }

    winpct_lookup = {}
    for _, r in win_pcts.iterrows():
        winpct_lookup[(int(r["Season"]), int(r["TeamID"]))] = r["win_pct"]

    return seeds_lookup, stats_lookup, winpct_lookup


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

        # Regular season games
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

        # Tournament games
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
    # Mark tournament games for CV splitting
    if "is_tourney" not in df.columns:
        df["is_tourney"] = 0
    df["is_tourney"] = df["is_tourney"].fillna(0).astype(int)
    return df


# ── 6. Feature columns ───────────────────────────────────────────────────────
def get_feature_cols(df):
    exclude = {"Season", "T1", "T2", "target", "weight", "is_tourney"}
    return [c for c in df.columns if c not in exclude]


# ── 7. Optuna Tuning + Training ──────────────────────────────────────────────
def run_optuna(train_df, feature_cols, n_trials=100):
    """Run Optuna hyperparameter search with expanding-window time-series CV."""
    val_seasons = [2022, 2023, 2024, 2025]

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "nthread": -1,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
        }

        fold_losses = []
        for i, val_season in enumerate(val_seasons):
            tr = train_df[train_df["Season"] < val_season]
            va = train_df[(train_df["Season"] == val_season) & (train_df["is_tourney"] == 1)]

            if len(va) == 0:
                continue

            X_tr = tr[feature_cols]
            y_tr = tr["target"]
            w_tr = tr["weight"]
            X_va = va[feature_cols]
            y_va = va["target"]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

            preds = model.predict_proba(X_va)[:, 1]
            preds = np.clip(preds, CLIP_LOW, CLIP_HIGH)
            ll = log_loss(y_va, preds)
            fold_losses.append(ll)

            # Optuna pruning
            trial.report(np.mean(fold_losses), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_losses)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n{'='*60}")
    print(f"Best trial log loss: {study.best_value:.6f}")
    print(f"Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Print per-fold results with best params
    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": -1,
        "verbosity": 0,
        **study.best_params,
    }
    print(f"\nPer-fold log loss with best params:")
    for val_season in val_seasons:
        tr = train_df[train_df["Season"] < val_season]
        va = train_df[(train_df["Season"] == val_season) & (train_df["is_tourney"] == 1)]
        if len(va) == 0:
            print(f"  {val_season}: no tournament games")
            continue
        model = xgb.XGBClassifier(**best_params)
        model.fit(tr[feature_cols], tr["target"], sample_weight=tr["weight"], verbose=False)
        preds = np.clip(model.predict_proba(va[feature_cols])[:, 1], CLIP_LOW, CLIP_HIGH)
        ll = log_loss(va["target"], preds)
        print(f"  {val_season}: {ll:.6f} ({len(va)} games)")

    print(f"{'='*60}\n")
    return study.best_params


def train_final_model(train_df, feature_cols, best_params):
    """Train final model on all available data using best hyperparameters."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "nthread": -1,
        "verbosity": 0,
        **best_params,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        train_df[feature_cols],
        train_df["target"],
        sample_weight=train_df["weight"],
        verbose=False,
    )
    return model


# ── 8. Generate Submission ────────────────────────────────────────────────────
def generate_submission(model, feature_cols, submission_template,
                        elo_m, elo_w, seeds_m, seeds_w, stats_m, stats_w, wp_m, wp_w):
    seeds_look_m, stats_look_m, wp_look_m = _make_lookups(seeds_m, stats_m, wp_m)
    seeds_look_w, stats_look_w, wp_look_w = _make_lookups(seeds_w, stats_w, wp_w)

    rows = []
    for _, r in submission_template.iterrows():
        parts = r["ID"].split("_")
        season, t1, t2 = int(parts[0]), int(parts[1]), int(parts[2])

        # Determine gender: men's teams are in 1000s, women's in 3000s
        is_mens = t1 < 3000
        elo_snap = elo_m if is_mens else elo_w
        seeds_look = seeds_look_m if is_mens else seeds_look_w
        stats_look = stats_look_m if is_mens else stats_look_w
        wp_look = wp_look_m if is_mens else wp_look_w

        row = _build_feature_row(season, t1, t2, elo_snap, seeds_look, stats_look, wp_look, is_mens)
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    X = pred_df[feature_cols]
    preds = model.predict_proba(X)[:, 1]
    preds = np.clip(preds, CLIP_LOW, CLIP_HIGH)

    submission = submission_template.copy()
    submission["Pred"] = preds
    return submission


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    data = load_data()

    print("Computing ELO ratings...")
    elo_m = compute_elo_ratings(data["mens_compact"], data["mens_tourney"])
    elo_w = compute_elo_ratings(data["womens_compact"], data["womens_tourney"])

    print("Computing season stats...")
    stats_m = compute_season_stats(data["mens_detailed"])
    stats_w = compute_season_stats(data["womens_detailed"])

    print("Parsing seeds and win percentages...")
    seeds_m = parse_seeds(data["mens_seeds"])
    seeds_w = parse_seeds(data["womens_seeds"])
    wp_m = compute_win_pct(data["mens_compact"])
    wp_w = compute_win_pct(data["womens_compact"])

    print("Building training data...")
    train_df = build_training_data(data, elo_m, elo_w, stats_m, stats_w, seeds_m, seeds_w, wp_m, wp_w)
    feature_cols = get_feature_cols(train_df)
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Tournament games: {train_df['is_tourney'].sum():,.0f}")

    print("\nRunning Optuna hyperparameter tuning (100 trials)...")
    best_params = run_optuna(train_df, feature_cols, n_trials=100)

    print("Training final model on all data...")
    model = train_final_model(train_df, feature_cols, best_params)

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)

    print("Generating submission...")
    submission = generate_submission(
        model, feature_cols, data["submission"],
        elo_m, elo_w, seeds_m, seeds_w, stats_m, stats_w, wp_m, wp_w,
    )
    submission.to_csv("submission.csv", index=False)

    # Verification
    print(f"\nSubmission shape: {submission.shape}")
    print(f"Expected rows: {len(data['submission'])}")
    print(f"Pred range: [{submission['Pred'].min():.4f}, {submission['Pred'].max():.4f}]")
    print(f"Pred mean: {submission['Pred'].mean():.4f}")
    print(f"Any exact 0 or 1: {((submission['Pred'] == 0) | (submission['Pred'] == 1)).any()}")
    print("\nDone! Submission saved to submission.csv")


if __name__ == "__main__":
    main()
