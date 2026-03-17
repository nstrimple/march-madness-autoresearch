# autoresearch — March Madness

Autonomous ML experimentation loop for NCAA tournament prediction.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar16`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `CLAUDE.md` — project overview and data layout.
   - `prepare.py` — fixed constants, data loading, season stats, evaluation harness. Do not modify.
   - `train.py` — the file you modify. ELO, features, model, hyperparameters.
4. **Verify data exists**: Check that CSV files exist in the project root (e.g. `MTeams.csv`, `MRegularSeasonDetailedResults.csv`). If not, tell the human to run `unzip march-machine-learning-mania-2026.zip`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Task

Predict NCAA Men's and Women's basketball tournament game outcomes. The goal is to minimize **log loss** on tournament games from 2022–2025 (expanding-window cross-validation). Lower is better. Predicting 0.5 for all games is the baseline (~0.693).

### Data available

All CSV files are in the project root directory:

- `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` — box-score stats per game
- `MRegularSeasonCompactResults.csv` / `WRegularSeasonCompactResults.csv` — W/L results
- `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` — historical tournament outcomes
- `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` — tournament seeding
- `MMasseyOrdinals.csv` — external ranking systems (~129 MB, now in use — avg + elite system ranks)
- `MTeamConferences.csv` / `WTeamConferences.csv` — conference membership
- `MTeamCoaches.csv` — coaching data
- `MGameCities.csv` / `WGameCities.csv` — game locations
- `Cities.csv` — city metadata
- `Conferences.csv` — conference names

## Experimentation

Each experiment runs on CPU. The training script runs for a **fixed pipeline** of data loading + feature engineering + XGBoost training + CV evaluation. You launch it simply as: `python train.py`.

Typical runtime: ~1-2 minutes. Time budget: 3 minutes max.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: ELO parameters, feature engineering, model choice (XGBoost, LightGBM, logistic regression, ensembles), hyperparameters, data weighting, new data sources from the CSVs.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, season stats, and constants.
- Modify `CLAUDE.md`.
- Modify `program.md` **except** for the experiment backlog section (you SHOULD update that after each experiment).
- Install new packages. You can only use what's already installed: pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, joblib.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_logloss.**

### Experiment backlog

This is a living list. After each experiment (keep or discard), reflect on what you learned, then **propose 1–3 new experiment ideas** and append them to the bottom of this backlog. Cross off or annotate ideas that have been tried. The goal is to always have a rich queue of untried ideas so you never stall.

#### Feature engineering
- [x] **Massey Ordinals — all-system average** — avg/median/best rank across 197 systems. ✅ kept
- [x] **Massey Ordinals — elite systems** — avg rank from POM, SAG, MOR, COL, BPI, RPI only. ✅ kept (0.490)
- [x] **Massey Ordinals — individual elite systems** — each as separate feature. ❌ hurt (0.495)
- [x] **Efficiency metrics** — off/def efficiency, eFG%, TO rate, OR rate, FT rate, tempo. ✅ kept
- [x] **Recent form** — last 10 games stats. ❌ hurt (0.498)
- [x] **Conference strength** — avg ELO of conference, SOS adjustments. ❌ hurt (0.497)
- [x] **Strength of schedule** — avg opponent ELO. ❌ hurt (0.496)
- [ ] **Coach tournament experience** — `MTeamCoaches.csv` — years coaching, prior tournament appearances, career tournament wins
- [x] **Seed × ELO interaction** — ❌ hurt (0.496)
- [ ] **Win streak going into tournament** — length of current win/loss streak at end of regular season
- [ ] **Road/neutral record** — win% in away/neutral games only (tournament is neutral site)
- [ ] **Close game performance** — win% in games decided by ≤5 points (poise under pressure)
- [ ] **Scoring variance** — std dev of points scored (consistency metric)
- [ ] **Massey rank trajectory** — rank improvement from mid-season to end-of-season (momentum)

#### Model architecture
- [x] **LightGBM defaults** — ❌ hurt (0.495)
- [x] **Ensemble: XGB + LightGBM 50/50 blend** — ❌ hurt (0.493)
- [ ] **LightGBM tuned** — match XGB-style params, tune num_leaves separately
- [ ] **Ensemble: XGB + LightGBM 70/30 blend** — XGB-heavy since it's stronger alone
- [ ] **Logistic regression** — on top ~10 features only (simple baseline to understand feature importance)
- [ ] **Ensemble: XGB + logistic regression blend** — combine tree model with linear model for diversity
- [ ] **Stacking** — train a meta-learner on out-of-fold predictions from XGB + LightGBM + LogReg
- [ ] **Neural network** — simple MLP via sklearn MLPClassifier on key features
- [x] **Eval on tourney + reg season** — ✅ kept (0.469, much broader signal)
- [ ] **Separate men's/women's models (with reg season eval)** — retry now that eval includes reg season

#### Hyperparameter tuning
- [x] **XGB depth=4 lr=0.03 n=800 + regularization** — ✅ kept
- [x] **XGB lower lr=0.01 n=1500** — ❌ hurt (0.493)
- [x] **XGB depth=3** — ❌ hurt (0.492)
- [ ] **XGB depth=5 or 6** — deeper trees to capture more interactions
- [ ] **XGB subsample=0.6 colsample=0.6** — more aggressive subsampling
- [x] **Early stopping** — ❌ hurt (0.495, lost training data)
- [x] **Separate men's/women's models** — ❌ hurt (0.496, not enough data per model)
- [ ] **Bayesian-style sweep** — try 5 random param combos and keep the best

#### Data strategy
- [x] **Tournament weight = 6.0** — ❌ hurt (0.496)
- [ ] **Tournament weight = 2.0** — less emphasis on tournament games in training
- [ ] **Recency weighting** — weight recent seasons (2018+) higher than older seasons
- [ ] **Drop pre-2010 data** — old seasons may have different dynamics, hurt more than help
- [ ] **Tournament-only training** — train exclusively on historical tournament games
- [ ] **Seed-based prior blending** — blend model output with historical seed-matchup win rates

#### ELO tuning
- [x] **K=32, season_regression=0.80** — ❌ hurt (0.504)
- [ ] **K=15, season_regression=0.80** — slower adaptation, more carryover
- [ ] **Remove home court bonus entirely** — tournament is neutral site, HCA may add noise
- [ ] **Margin-of-victory cap = 1.5** — reduce influence of blowouts on ELO

#### Radical / creative
- [x] **Use only diff features + key absolutes** — ❌ hurt (0.494)
- [ ] **Feature selection via importance** — run one model, drop bottom 50% features by importance, retrain
- [ ] **Prediction calibration** — apply Platt scaling or isotonic regression to final predictions
- [ ] **Target smoothing** — use soft labels (0.1/0.9) instead of hard (0/1) for training
- [ ] **Matchup-style features** — fast team vs slow team (tempo diff × eFG diff interactions)
- [ ] **Conference tournament results** — `MConferenceTourneyGames.csv` — how team performed in conf tourney
- [ ] **Re-try features that hurt before** — now with reg season eval, previously bad features might help (recent form, SOS, interactions)
- [ ] **Weighted reg season eval** — weight tournament game predictions higher in the combined eval
- [ ] **Logistic regression on reg season** — simple model may do well on predictable reg season games

### Simplicity criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_logloss:      0.XXXXXX
per_fold:         2022=0.XX 2023=0.XX 2024=0.XX 2025=0.XX
total_seconds:    XX.X
```

You can extract the key metric from the log file:

```
grep "^val_logloss:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	val_logloss	status	description
```

1. git commit hash (short, 7 chars)
2. val_logloss achieved (e.g. 0.523456) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	val_logloss	status	description
a1b2c3d	0.523456	keep	baseline
b2c3d4e	0.518200	keep	increase K factor to 32
c3d4e5f	0.530000	discard	switch to LightGBM default params
d4e5f6g	0.000000	crash	added Massey ordinals (KeyError on merge)
e5f6g7h	0.512100	keep	add Massey ordinals (top 5 systems averaged)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar16`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. **Pick an experiment** from the backlog in the "Experiment backlog" section above. Prefer untried ideas. Consider what you've learned from prior results to guide your choice.
3. Tune `train.py` with the experimental idea by directly hacking the code.
4. git commit
5. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^val_logloss:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If val_logloss improved (lower), you "advance" the branch, keeping the git commit
10. If val_logloss is equal or worse, you git reset back to where you started
11. **Reflect and propose**: Based on the result (and all prior results), think about what worked or didn't and why. Propose 1–3 new experiment ideas and add them to the backlog in `program.md`. Mark the experiment you just ran as tried. This keeps the idea pipeline fresh.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~2 minutes total. If a run exceeds 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (a bug, missing column, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the data files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~2 minutes then you can run approx 30/hour. The user then wakes up to experimental results, all completed by you while they slept!
