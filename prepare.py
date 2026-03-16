"""
Immutable data loading and evaluation infrastructure for March Madness predictions.
DO NOT MODIFY — the autoresearch agent only modifies train.py.

Usage:
    python prepare.py              # verify data loads correctly
    import prepare                 # use from train.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# ── Constants (fixed, do not modify) ─────────────────────────────────────────
DATA_DIR = ""
CLIP_LOW = 0.05
CLIP_HIGH = 0.95
VAL_SEASONS = [2022, 2023, 2024, 2025]
TIME_BUDGET = 180  # seconds

STAT_COLS = [
    "fg_pct", "fg3_pct", "ft_pct", "ppg", "opp_ppg", "score_diff",
    "or_pg", "dr_pg", "ast_pg", "to_pg", "ast_to_ratio",
    "stl_pg", "blk_pg", "opp_fg_pct", "opp_fg3_pct", "opp_to_pg",
]


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_data():
    """Load all CSV data files for men's and women's basketball."""
    data = {}
    for prefix, label in [("M", "mens"), ("W", "womens")]:
        data[f"{label}_compact"] = pd.read_csv(f"{DATA_DIR}{prefix}RegularSeasonCompactResults.csv")
        data[f"{label}_detailed"] = pd.read_csv(f"{DATA_DIR}{prefix}RegularSeasonDetailedResults.csv")
        data[f"{label}_tourney"] = pd.read_csv(f"{DATA_DIR}{prefix}NCAATourneyCompactResults.csv")
        data[f"{label}_seeds"] = pd.read_csv(f"{DATA_DIR}{prefix}NCAATourneySeeds.csv")
    data["submission"] = pd.read_csv(f"{DATA_DIR}SampleSubmissionStage1.csv")
    return data


# ── Season Stats ─────────────────────────────────────────────────────────────
def _team_game_stats(g, is_winner):
    """Extract per-team stats from a single game row."""
    pf = "W" if is_winner else "L"
    op = "L" if is_winner else "W"
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


def compute_season_stats(detailed_results):
    """Compute per-team per-season average box-score stats from detailed results."""
    rows = []
    for _, g in detailed_results.iterrows():
        rows.append(_team_game_stats(g, is_winner=True))
        rows.append(_team_game_stats(g, is_winner=False))

    df = pd.DataFrame(rows)
    agg = df.groupby(["Season", "TeamID"]).mean(numeric_only=True).reset_index()
    return agg


# ── Seeds & Win % ────────────────────────────────────────────────────────────
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


# ── Lookups ──────────────────────────────────────────────────────────────────
def _make_lookups(seeds_parsed, season_stats, win_pcts):
    """Build dict lookups from DataFrames for fast feature construction."""
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


# ── Evaluation (DO NOT CHANGE — this is the fixed metric) ───────────────────
def evaluate(results):
    """
    Fixed evaluation harness for the autoresearch loop.

    Args:
        results: dict {season: (y_true, y_pred)} from train_and_predict()

    Returns:
        (mean_logloss, fold_dict) where fold_dict maps season -> logloss
    """
    fold_losses = {}
    for season in VAL_SEASONS:
        if season not in results:
            continue
        y_true, y_pred = results[season]
        y_pred_clipped = np.clip(y_pred, CLIP_LOW, CLIP_HIGH)
        ll = log_loss(y_true, y_pred_clipped)
        fold_losses[season] = ll

    if not fold_losses:
        return float("inf"), {}

    mean_ll = float(np.mean(list(fold_losses.values())))
    return mean_ll, fold_losses


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print("Data loaded successfully.\n")
    for key in sorted(data.keys()):
        df = data[key]
        print(f"  {key:30s} {str(df.shape):>15s}")

    print("\nComputing season stats (mens)...")
    stats = compute_season_stats(data["mens_detailed"])
    print(f"  Shape: {stats.shape}")

    print("Parsing seeds (mens)...")
    seeds = parse_seeds(data["mens_seeds"])
    print(f"  Shape: {seeds.shape}")

    print("Computing win pct (mens)...")
    wp = compute_win_pct(data["mens_compact"])
    print(f"  Shape: {wp.shape}")

    print("\nAll prepare.py functions working.")
