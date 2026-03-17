"""
NCAA Tournament bracket prediction using the full trained model pipeline.
Uses XGB+LGB ensemble with seed prior blending.

Usage: python predict.py                    # Men's 2026
       python predict.py --gender both      # Both genders
       python predict.py --model-path m.joblib --season 2025
"""

import argparse
import os

import numpy as np
import pandas as pd
import joblib

from prepare import (
    load_data, compute_season_stats, parse_seeds, compute_win_pct,
    _make_lookups, CLIP_LOW, CLIP_HIGH,
)
from train import (
    compute_elo_ratings, compute_efficiency_stats, compute_massey_ranks,
    compute_road_record, compute_close_game_record, compute_conf_tourney_record,
    compute_win_streak, compute_power_conf, compute_last_n_stats,
    _build_feature_row, _make_efficiency_lookup,
    MODEL_PATH, EFFICIENCY_COLS, LAST_N_COLS,
)
from inference import (
    load_bracket_data, simulate_bracket as _sim_bracket_base,
    render_bracket, render_game, resolve_ref, extract_seed_num,
    format_team, format_bar, ROUND_NAMES, REGIONS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict NCAA tournament bracket (full pipeline)")
    parser.add_argument("--gender", choices=["mens", "womens", "both"], default="mens")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    return parser.parse_args()


def load_model(model_path):
    """Load saved model dict from train.py --save."""
    saved = joblib.load(model_path)
    # Validate it's the full pipeline format
    if "xgb_model" not in saved:
        raise ValueError(
            f"{model_path} is not a full-pipeline model. "
            "Run 'python train.py --save' first."
        )
    return saved


def build_all_lookups(data):
    """Compute all feature lookups for both genders."""
    print("Computing ELO ratings...")
    elo_m = compute_elo_ratings(data["mens_compact"], data["mens_tourney"])
    elo_w = compute_elo_ratings(data["womens_compact"], data["womens_tourney"])

    print("Computing season stats...")
    stats_m = compute_season_stats(data["mens_detailed"])
    stats_w = compute_season_stats(data["womens_detailed"])

    print("Parsing seeds and win pct...")
    seeds_m = parse_seeds(data["mens_seeds"])
    seeds_w = parse_seeds(data["womens_seeds"])
    wp_m = compute_win_pct(data["mens_compact"])
    wp_w = compute_win_pct(data["womens_compact"])

    print("Loading Massey ordinals...")
    massey_lookup = compute_massey_ranks()

    print("Computing efficiency stats...")
    eff_m = _make_efficiency_lookup(compute_efficiency_stats(data["mens_detailed"]))
    eff_w = _make_efficiency_lookup(compute_efficiency_stats(data["womens_detailed"]))

    print("Computing road/neutral records...")
    road_m = compute_road_record(data["mens_compact"])
    road_w = compute_road_record(data["womens_compact"])

    print("Computing close game records...")
    close_m = compute_close_game_record(data["mens_compact"])
    close_w = compute_close_game_record(data["womens_compact"])

    print("Computing conference tournament results...")
    conf_m, conf_w = compute_conf_tourney_record()

    print("Computing win streaks...")
    streak_m = compute_win_streak(data["mens_compact"])
    streak_w = compute_win_streak(data["womens_compact"])

    print("Computing power conference indicators...")
    power_m = compute_power_conf("M")
    power_w = compute_power_conf("W")

    print("Computing last-N game stats...")
    last_n_m = compute_last_n_stats(data["mens_detailed"])
    last_n_w = compute_last_n_stats(data["womens_detailed"])

    # Build standard lookups
    seeds_look_m, stats_look_m, wp_look_m = _make_lookups(seeds_m, stats_m, wp_m)
    seeds_look_w, stats_look_w, wp_look_w = _make_lookups(seeds_w, stats_w, wp_w)

    lookups_m = {
        "elo": elo_m,
        "seeds": seeds_look_m,
        "stats": stats_look_m,
        "winpct": wp_look_m,
        "massey": massey_lookup,
        "eff": eff_m,
        "road": road_m,
        "close": close_m,
        "conf": conf_m,
        "streak": streak_m,
        "power": power_m,
        "last_n": last_n_m,
    }
    lookups_w = {
        "elo": elo_w,
        "seeds": seeds_look_w,
        "stats": stats_look_w,
        "winpct": wp_look_w,
        "massey": massey_lookup,
        "eff": eff_w,
        "road": road_w,
        "close": close_w,
        "conf": conf_w,
        "streak": streak_w,
        "power": power_w,
        "last_n": last_n_w,
    }
    return lookups_m, lookups_w


def predict_game(t1, t2, saved, lookups, season, is_mens):
    """Ensemble prediction with seed prior blending."""
    row = _build_feature_row(
        season, t1, t2,
        lookups["elo"], lookups["seeds"], lookups["stats"], lookups["winpct"],
        is_mens,
        massey_lookup=lookups["massey"],
        eff_lookup=lookups["eff"],
        road_lookup=lookups["road"],
        close_lookup=lookups["close"],
        conf_tourney_lookup=lookups["conf"],
        streak_lookup=lookups["streak"],
        power_lookup=lookups["power"],
        last_n_lookup=lookups["last_n"],
    )
    df = pd.DataFrame([row])
    feature_cols = saved["feature_cols"]

    # Add missing columns as NaN
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    xgb_prob = saved["xgb_model"].predict_proba(df[feature_cols])[0][1]
    lgb_prob = saved["lgb_model"].predict_proba(df[feature_cols])[0][1]

    xgb_weight = saved["xgb_weight"]
    prob = xgb_weight * xgb_prob + (1 - xgb_weight) * lgb_prob

    # Seed prior blending
    s1 = lookups["seeds"].get((season, t1), None)
    s2 = lookups["seeds"].get((season, t2), None)
    if s1 is not None and s2 is not None and not (pd.isna(s1) or pd.isna(s2)):
        prior = saved["seed_prior"].get((int(s1), int(s2)), 0.5)
        alpha = saved["blend_alpha"]
        prob = (1 - alpha) * prob + alpha * prior

    return float(np.clip(prob, CLIP_LOW, CLIP_HIGH))


def simulate_bracket(slots_df, seeds_raw, season, saved, lookups, is_mens):
    """Simulate the full tournament bracket using ensemble predictions."""
    seed_to_team = {}
    team_seed_str = {}
    for _, row in seeds_raw.iterrows():
        seed_str = row["Seed"]
        team_id = int(row["TeamID"])
        seed_to_team[seed_str] = team_id
        team_seed_str[team_id] = seed_str

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

        t1, t2 = min(team_a, team_b), max(team_a, team_b)
        prob_t1 = predict_game(t1, t2, saved, lookups, season, is_mens)

        if prob_t1 >= 0.5:
            winner = t1
            winner_prob = prob_t1
        else:
            winner = t2
            winner_prob = 1.0 - prob_t1

        seed_a = team_seed_str.get(team_a, strong)
        seed_b = team_seed_str.get(team_b, weak)

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


def main():
    args = parse_args()
    season = args.season

    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} not found.")
        print("Run 'python train.py --save' first to train and save the model.")
        return

    print(f"Loading model from {args.model_path}...")
    saved = load_model(args.model_path)
    print(f"  Features: {len(saved['feature_cols'])} (selected from {len(saved['all_feature_cols'])})")
    print(f"  Ensemble: {saved['xgb_weight']:.0%} XGB + {1-saved['xgb_weight']:.0%} LGB")
    print(f"  Seed prior blend: {saved['blend_alpha']:.0%}")

    print("\nLoading data...")
    data = load_data()

    print("Building feature lookups...")
    lookups_m, lookups_w = build_all_lookups(data)

    genders = []
    if args.gender in ("mens", "both"):
        genders.append(("mens", "M", True))
    if args.gender in ("womens", "both"):
        genders.append(("womens", "W", False))

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
            slots, seeds_raw, season, saved, lookups, is_mens
        )

        render_bracket(
            slot_results, team_seed_str, team_names, region_names, gender_label, season
        )
        print(f"  Total games simulated: {len(slot_results)}")


if __name__ == "__main__":
    main()
