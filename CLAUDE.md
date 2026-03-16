# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition: **March Machine Learning Mania 2026** — predict NCAA Men's and Women's basketball tournament game outcomes. The goal is to submit win probabilities for all possible matchups.

Dataset is in `march-machine-learning-mania-2026.zip` (extract to a `data/` directory before working with files).

## Data Layout

All CSV files use a `M` prefix for Men's data and `W` prefix for Women's data.

Key files:
- `MTeams.csv` / `WTeams.csv` — team IDs and names (master lookup)
- `MSeasons.csv` / `WSeasons.csv` — season metadata (DayZero for date math)
- `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` — box-score stats per game
- `MNCAATourneyCompactResults.csv` / `WNCAATourneyCompactResults.csv` — historical tournament outcomes
- `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` — tournament seeding by season
- `MMasseyOrdinals.csv` — external ranking systems (large file, ~129 MB)
- `SampleSubmissionStage1.csv` / `SampleSubmissionStage2.csv` — required output format

Submission format: rows keyed as `Season_Team1ID_Team2ID` (Team1ID < Team2ID always), with a `Pred` column for win probability of Team1.

## Competition Stages

- **Stage 1**: Predict all possible matchups from 2021–2025 seasons (historical evaluation)
- **Stage 2**: Predict the actual 2026 tournament bracket (released during tournament)

## Development Setup

```bash
# Extract data
unzip march-machine-learning-mania-2026.zip -d data/

# Install common dependencies (no requirements.txt yet)
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn jupyter
```

## Common Patterns

**Loading data:**
```python
import pandas as pd
DATA_DIR = "data/"
teams = pd.read_csv(f"{DATA_DIR}MTeams.csv")
results = pd.read_csv(f"{DATA_DIR}MRegularSeasonDetailedResults.csv")
```

**Submission generation** must cover every `Season_T1_T2` combination in the sample submission file — not just tournament matchups.

**Score metric**: Log loss (lower is better). Predicting 0.5 for all games is the baseline.
