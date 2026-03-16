"""
NCAA Tournament bracket simulation and CLI rendering.
Uses the trained XGBoost model from march_madness.py to predict game-by-game outcomes.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import joblib

from prepare import (
    load_data, compute_season_stats, parse_seeds, compute_win_pct,
    _make_lookups, CLIP_LOW, CLIP_HIGH, DATA_DIR,
)
from train import (
    compute_elo_ratings, _build_feature_row, build_training_data,
    get_feature_cols, train_final_model, MODEL_PATH,
)
from march_madness import run_optuna

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
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

ROUND_NAMES = {
    "R1": "ROUND OF 64",
    "R2": "ROUND OF 32",
    "R3": "SWEET 16",
    "R4": "ELITE 8",
    "R5": "FINAL FOUR",
    "R6": "CHAMPIONSHIP",
}

REGIONS = ["W", "X", "Y", "Z"]


# ── CLI Arguments ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate NCAA tournament bracket")
    parser.add_argument("--gender", choices=["mens", "womens", "both"], default="mens")
    parser.add_argument("--skip-optuna", action="store_true", help="Use default params")
    parser.add_argument("--params-file", type=str, help="Load params from JSON")
    parser.add_argument("--save-params", type=str, help="Save params to JSON")
    parser.add_argument("--save-model", type=str, help="Save model via joblib")
    parser.add_argument("--load-model", type=str, default=MODEL_PATH, help="Load model via joblib (default: model.joblib)")
    parser.add_argument("--season", type=int, default=2026)
    return parser.parse_args()


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_bracket_data(season, prefix):
    """Load bracket-specific data for a gender prefix (M or W)."""
    slots = pd.read_csv(f"{DATA_DIR}{prefix}NCAATourneySlots.csv")
    slots = slots[slots["Season"] == season]

    seeds_raw = pd.read_csv(f"{DATA_DIR}{prefix}NCAATourneySeeds.csv")
    seeds_raw = seeds_raw[seeds_raw["Season"] == season]

    teams = pd.read_csv(f"{DATA_DIR}{prefix}Teams.csv")
    team_names = dict(zip(teams["TeamID"], teams["TeamName"]))

    seasons = pd.read_csv(f"{DATA_DIR}{prefix}Seasons.csv")
    season_row = seasons[seasons["Season"] == season]
    region_names = {}
    if len(season_row) > 0:
        row = season_row.iloc[0]
        for letter in REGIONS:
            region_names[letter] = row[f"Region{letter}"]

    return seeds_raw, slots, team_names, region_names


# ── Bracket Simulation ───────────────────────────────────────────────────────

def resolve_ref(ref, seed_to_team, slot_results):
    """Resolve a seed/slot reference to a team ID."""
    if ref in slot_results:
        return slot_results[ref]["winner"]
    if ref in seed_to_team:
        return seed_to_team[ref]
    raise ValueError(f"Cannot resolve reference: {ref}")


def predict_game(t1, t2, model, feature_cols, lookups, season, is_mens):
    """Predict P(lower-ID team wins) for a matchup."""
    elo_snapshot, seeds_lookup, stats_lookup, winpct_lookup = lookups
    row = _build_feature_row(
        season, t1, t2, elo_snapshot, seeds_lookup, stats_lookup, winpct_lookup, is_mens
    )
    df = pd.DataFrame([row])
    prob = model.predict_proba(df[feature_cols])[0][1]
    return float(np.clip(prob, CLIP_LOW, CLIP_HIGH))


def simulate_bracket(slots_df, seeds_raw, season, model, feature_cols, lookups, is_mens):
    """Simulate the full tournament bracket, returning results for each slot."""
    # Build seed_to_team mapping
    seed_to_team = {}
    team_seed_str = {}
    for _, row in seeds_raw.iterrows():
        seed_str = row["Seed"]
        team_id = int(row["TeamID"])
        seed_to_team[seed_str] = team_id
        team_seed_str[team_id] = seed_str

    # Sort slots: play-in first (no R prefix), then R1, R2, ..., R6
    def slot_sort_key(slot):
        if slot.startswith("R"):
            round_num = int(slot[1])
            return (1, round_num, slot)
        return (0, 0, slot)

    slots_sorted = slots_df.sort_values(
        "Slot", key=lambda s: s.map(slot_sort_key)
    ).reset_index(drop=True)

    slot_results = {}

    for _, row in slots_sorted.iterrows():
        slot = row["Slot"]
        strong = row["StrongSeed"]
        weak = row["WeakSeed"]

        team_a = resolve_ref(strong, seed_to_team, slot_results)
        team_b = resolve_ref(weak, seed_to_team, slot_results)

        # Model convention: t1 < t2
        t1, t2 = min(team_a, team_b), max(team_a, team_b)
        prob_t1 = predict_game(t1, t2, model, feature_cols, lookups, season, is_mens)

        if prob_t1 >= 0.5:
            winner = t1
            winner_prob = prob_t1
        else:
            winner = t2
            winner_prob = 1.0 - prob_t1

        # Get seed strings for display
        seed_a = team_seed_str.get(team_a, strong)
        seed_b = team_seed_str.get(team_b, weak)

        # For play-in winners, assign the base seed string
        if not slot.startswith("R"):
            team_seed_str[winner] = slot

        slot_results[slot] = {
            "team_a": team_a,
            "team_b": team_b,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "winner": winner,
            "winner_prob": winner_prob,
            "prob_t1": prob_t1,
            "t1": t1,
            "t2": t2,
        }

    return slot_results, team_seed_str


# ── CLI Rendering ────────────────────────────────────────────────────────────

def extract_seed_num(seed_str):
    """Extract numeric seed from seed string (e.g., 'W01' -> 1, 'X16a' -> 16)."""
    digits = "".join(filter(str.isdigit, seed_str))
    return int(digits) if digits else 0


def format_team(team_id, seed_str, team_names, width=18):
    """Format a team line: (seed) Name."""
    seed_num = extract_seed_num(seed_str)
    name = team_names.get(team_id, f"Team {team_id}")
    label = f"({seed_num}) {name}"
    if len(label) > width:
        label = label[:width]
    return label.ljust(width)


def format_bar(prob, width=10):
    """Create a probability bar like ▓▓▓▓▓▓▓▓░░."""
    filled = round(prob * width)
    return "\u2593" * filled + "\u2591" * (width - filled)


def render_game(result, team_names, team_seed_str):
    """Render a single game as 2 lines."""
    team_a = result["team_a"]
    team_b = result["team_b"]
    seed_a = team_seed_str.get(team_a, result["seed_a"])
    seed_b = team_seed_str.get(team_b, result["seed_b"])
    winner = result["winner"]

    # Compute display probabilities
    t1, t2 = result["t1"], result["t2"]
    prob_t1 = result["prob_t1"]
    prob_a = prob_t1 if team_a == t1 else 1.0 - prob_t1
    prob_b = 1.0 - prob_a

    label_a = format_team(team_a, seed_a, team_names)
    label_b = format_team(team_b, seed_b, team_names)
    bar_a = format_bar(prob_a)
    bar_b = format_bar(prob_b)

    marker_a = " \u2190" if winner == team_a else ""
    marker_b = " \u2190" if winner == team_b else ""

    line_a = f"  {label_a} {bar_a} {prob_a:>4.0%}{marker_a}"
    line_b = f"  {label_b} {bar_b} {prob_b:>4.0%}{marker_b}"

    return line_a, line_b


def render_bracket(slot_results, team_seed_str, team_names, region_names, gender_label, season):
    """Render the full bracket to stdout."""
    label = gender_label.upper()
    header = f" NCAA {label} TOURNAMENT {season} "
    print(f"\n{'=' * 20}{header}{'=' * 20}")

    # Categorize slots by round and region
    playin_slots = []
    round_region_slots = {}  # (round_prefix, region_letter) -> [slots]
    final_slots = {}  # round_prefix -> [slots]

    for slot in slot_results:
        if not slot.startswith("R"):
            playin_slots.append(slot)
        elif slot.startswith("R5") or slot.startswith("R6"):
            rnd = slot[:2]
            final_slots.setdefault(rnd, []).append(slot)
        else:
            rnd = slot[:2]
            region = slot[2] if len(slot) > 2 else ""
            round_region_slots.setdefault((rnd, region), []).append(slot)

    # Play-in games
    if playin_slots:
        print(f"\n-- FIRST FOUR {'-' * 46}")
        for slot in sorted(playin_slots):
            result = slot_results[slot]
            line_a, line_b = render_game(result, team_names, team_seed_str)
            print(line_a)
            print(line_b)
            print()

    # Rounds 1-4 by region
    for rnd_prefix in ["R1", "R2", "R3", "R4"]:
        rnd_name = ROUND_NAMES.get(rnd_prefix, rnd_prefix)
        for region in REGIONS:
            key = (rnd_prefix, region)
            if key not in round_region_slots:
                continue
            region_name = region_names.get(region, region)
            dash_count = max(1, 48 - len(rnd_name) - len(region_name))
            print(f"-- {rnd_name} \u00b7 {region_name} {'-' * dash_count}")
            slots = sorted(round_region_slots[key])
            for slot in slots:
                result = slot_results[slot]
                line_a, line_b = render_game(result, team_names, team_seed_str)
                print(line_a)
                print(line_b)
                print()

    # Final Four
    if "R5" in final_slots:
        print(f"-- FINAL FOUR {'-' * 46}")
        for slot in sorted(final_slots["R5"]):
            result = slot_results[slot]
            line_a, line_b = render_game(result, team_names, team_seed_str)
            print(line_a)
            print(line_b)
            print()

    # Championship
    if "R6" in final_slots:
        print(f"-- CHAMPIONSHIP {'-' * 44}")
        for slot in sorted(final_slots["R6"]):
            result = slot_results[slot]
            line_a, line_b = render_game(result, team_names, team_seed_str)
            print(line_a)
            print(line_b)
            print()

        # Show predicted champion
        champ_slot = sorted(final_slots["R6"])[0]
        champ_id = slot_results[champ_slot]["winner"]
        champ_seed = team_seed_str.get(champ_id, "??")
        champ_name = team_names.get(champ_id, f"Team {champ_id}")
        seed_num = extract_seed_num(champ_seed)
        print(f"  PREDICTED CHAMPION: ({seed_num}) {champ_name}")

    print()


# ── Model Preparation ────────────────────────────────────────────────────────

def prepare_model(args, data, elo_m, elo_w, stats_m, stats_w, seeds_m, seeds_w, wp_m, wp_w):
    """Prepare the model and feature columns."""
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}...")
        saved = joblib.load(args.load_model)
        return saved["model"], saved["feature_cols"]
    elif args.load_model and not os.path.exists(args.load_model):
        print(f"Model file {args.load_model} not found, training from scratch...")

    print("Building training data...")
    train_df = build_training_data(
        data, elo_m, elo_w, stats_m, stats_w, seeds_m, seeds_w, wp_m, wp_w
    )
    feature_cols = get_feature_cols(train_df)
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Features: {len(feature_cols)}")

    if args.params_file:
        print(f"Loading params from {args.params_file}...")
        with open(args.params_file) as f:
            best_params = json.load(f)
    elif args.skip_optuna:
        print("Using default parameters (--skip-optuna)...")
        best_params = DEFAULT_PARAMS.copy()
    else:
        print("Running Optuna hyperparameter tuning...")
        best_params = run_optuna(train_df, feature_cols)

    if args.save_params:
        print(f"Saving params to {args.save_params}...")
        with open(args.save_params, "w") as f:
            json.dump(best_params, f, indent=2)

    print("Training final model...")
    model = train_final_model(train_df, feature_cols, best_params)

    if args.save_model:
        print(f"Saving model to {args.save_model}...")
        joblib.dump({"model": model, "feature_cols": feature_cols}, args.save_model)

    return model, feature_cols


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    season = args.season

    genders = []
    if args.gender in ("mens", "both"):
        genders.append(("mens", "M", True))
    if args.gender in ("womens", "both"):
        genders.append(("womens", "W", False))

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

    model, feature_cols = prepare_model(
        args, data, elo_m, elo_w, stats_m, stats_w, seeds_m, seeds_w, wp_m, wp_w
    )

    # Build lookups for both genders
    lookups_m = (elo_m, *_make_lookups(seeds_m, stats_m, wp_m))
    lookups_w = (elo_w, *_make_lookups(seeds_w, stats_w, wp_w))

    for gender_label, prefix, is_mens in genders:
        print(f"\nSimulating {gender_label} bracket for {season}...")
        seeds_raw, slots, team_names, region_names = load_bracket_data(season, prefix)

        if len(slots) == 0:
            print(f"  No bracket data found for {season} {gender_label}")
            continue
        if len(seeds_raw) == 0:
            print(f"  No seeds found for {season} {gender_label}")
            continue

        lookups = lookups_m if is_mens else lookups_w
        slot_results, team_seed_str = simulate_bracket(
            slots, seeds_raw, season, model, feature_cols, lookups, is_mens
        )

        render_bracket(
            slot_results, team_seed_str, team_names, region_names, gender_label, season
        )
        print(f"  Total games simulated: {len(slot_results)}")


if __name__ == "__main__":
    main()
