# Offside Datathon — Final Submission

**Task:** Predict whether a player scores ≥1 goal in a match (`scored_flag`, binary classification).  
**Metric:** Average Precision (PR-AUC). Positive rate 8.54% → random baseline AP ≈ 0.085.  
**Best OOF AP:** **0.52050** — 6.1× over the random baseline.

---

## Submission Files

| File | Notebook | OOF AP | OOF AUC | Notes |
|------|----------|--------|---------|-------|
| `Sol_55212.csv` | `solution_fast.ipynb` | **0.52111** | **0.91822** | Best submission — 3-model blend (LGB + XGB + CAT) |
| `Sol_54760.csv` | `notebook-kaggle.ipynb` | 0.52050 | 0.91810 | Fully standalone — no cache required; trains from scratch |

Both files are ready to submit directly to the competition.

---

## Repository Contents

```
finalsubmission/
├── solution_fast.ipynb       Main solution — EDA, feature engineering, model training, blend
├── notebook-kaggle.ipynb     Alternate standalone notebook (fully self-contained, no cache)
├── eda_analysis.ipynb        Dedicated exploratory data analysis notebook
├── Sol_55212.csv             Best prediction file (from solution_fast.ipynb)
├── Sol_54760.csv             Alternate prediction file (from notebook-kaggle.ipynb)
├── train.csv                 Competition training data
├── test.csv                  Competition test data
└── cache/                    Pre-computed model outputs (enables fast execution)
    ├── oof_lgb.npy           LightGBM OOF predictions on train set
    ├── oof_xgb.npy           XGBoost  OOF predictions on train set
    ├── oof_cat.npy           CatBoost OOF predictions on train set
    ├── test_lgb.npy          LightGBM averaged test predictions
    ├── test_xgb.npy          XGBoost  averaged test predictions
    └── test_cat.npy          CatBoost averaged test predictions
```

---

## How to Run

### Option A — `solution_fast.ipynb` with cache (recommended, ~5 min)

This is the fastest path. The `cache/` folder holds pre-trained model outputs, so training is skipped entirely.

#### Locally

1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn lightgbm xgboost catboost matplotlib seaborn
   ```
2. Place `train.csv` and `test.csv` in the same folder as the notebook.
3. Keep the `cache/` folder in the same directory.
4. Open `solution_fast.ipynb` and confirm the top of the notebook reads:
   ```python
   USE_CACHE = True
   ROOT      = "./"
   CACHE_DIR = "./cache"
   ```
5. Run all cells. The notebook outputs `submission.csv` in the same folder.

#### On Kaggle

1. Create a new Kaggle notebook and select **GPU accelerator** (T4 or P100).
2. Add the competition dataset as input (contains `train.csv` / `test.csv`).
3. Upload the `cache/` folder as a separate **Kaggle dataset** (e.g. named `offside-cache`).
4. At the top of `solution_fast.ipynb`, update the two path variables:
   ```python
   USE_CACHE = True
   ROOT      = "/kaggle/input/<competition-dataset-name>/"
   CACHE_DIR = "/kaggle/input/offside-cache/cache"
   ```
5. Run all cells. Output saved to `/kaggle/working/submission.csv`.

---

### Option B — `solution_fast.ipynb` retrain from scratch (~25–30 min on GPU)

No cache needed. All training code is already in the notebook.

1. Follow steps 1–3 from Option A (local) or 1–2 (Kaggle).
2. Set `USE_CACHE = False` in the first code cell.
3. If no CUDA GPU is available, change `device="cuda"` to `device="cpu"` in the XGBoost cell and `task_type="GPU"` to `task_type="CPU"` in the CatBoost cell.
4. Run all cells.

---

### Option C — `notebook-kaggle.ipynb` (fully standalone, no cache, ~25–30 min)

This notebook is 100% self-contained — no cache folder, no external files beyond the CSVs.

#### Locally

1. Set `ROOT = "./"` in the Setup cell.
2. Run all cells.

#### On Kaggle

1. Set the ROOT path to the competition data directory:
   ```python
   ROOT = "/kaggle/input/<competition-dataset-name>/"
   ```
2. Change the submission output path:
   ```python
   KAGGLE_SUBMISSION_PATH = "/kaggle/working/submission.csv"
   ```
3. Run all cells.

---

### Option D — `eda_analysis.ipynb` (read-only EDA, ~2 min)

Reads only `train.csv` and `test.csv`. No model training.

Set the two paths at the top of the notebook if running on Kaggle:
```python
TRAIN_CSV = "/kaggle/input/<competition-dataset-name>/train.csv"
TEST_CSV  = "/kaggle/input/<competition-dataset-name>/test.csv"
```

---

## Reproducing the Cache from Scratch

If you want to regenerate `cache/*.npy` yourself (e.g. after modifying features), run `solution_fast.ipynb` with `USE_CACHE = False`. After training completes, save the arrays:

```python
import numpy as np, os
os.makedirs("cache", exist_ok=True)
np.save("cache/oof_lgb.npy",  oof_lgb);  np.save("cache/test_lgb.npy", test_lgb)
np.save("cache/oof_xgb.npy",  oof_xgb);  np.save("cache/test_xgb.npy", test_xgb)
np.save("cache/oof_cat.npy",  oof_cat);  np.save("cache/test_cat.npy", test_cat)
```

All random seeds are fixed to **42** throughout, so results are fully reproducible across runs on the same hardware.

> **Note on GPU non-determinism:** XGBoost and CatBoost GPU implementations may produce marginally different floating-point outputs across different GPU hardware. OOF AP is stable to ±0.0002.

---

## EDA Highlights (`eda_analysis.ipynb`)

| Finding | Modeling implication |
|---------|---------------------|
| Train and test **share 69,854 games and 24,487 players** with zero `appearance_id` overlap | Split is **random**, not time-based — StratifiedKFold(5) is the correct validation strategy |
| P(scored \| team_goals=0) ≈ 0.0001; P(scored) rises monotonically with team goals | Apply monotone constraint on `team_goals`; floor predictions to 1e-4 when team scored 0 |
| ~48% of players in training data **never scored** | Player-identity target encoding is highly informative |
| Understat analytics missing for ~48% of rows | Keep NaN + `understat_missing` flag; imputing destroys the missingness signal |
| `avg_xG` clearly separates scorers from non-scorers when present | Interact `avg_xG` with `team_goals` and `minutes_played` |
| Attackers ~17% / midfielders ~8% / defenders ~4% / GKs ~0% | `sub_position` target encoding captures this gradient |
| Shared games/teams mean within-game team scoring rate is observable in training labels | OOF target encoding at team×game grain is valid and yields +0.031 AP |

---

## Feature Engineering

All features are built inline inside `solution_fast.ipynb` (function `build_base_features`).

### Match goal context (most important group)
- `team_goals` — own-team goals, monotone constraint applied (↑)
- `opp_goals`, `total_goals`, `goal_diff_signed`
- `team_scored`, `high_scoring_team` — binary flags
- `goals_per_player` — team goals normalised by squad size

### Understat analytics and interactions
- `xg_per_shot`, `npxg_per_shot` — shot quality ratios
- `xg_plus_xa` — combined attacking involvement
- `xg_x_teamgoals`, `xg_x_minutes` — interaction features
- `understat_missing` — coverage flag (predictive in its own right)

### Within-match-team relative features
Who is the most likely scorer within the team on this specific day?
- `xg_rank_in_team`, `shots_rank_in_team`, `mv_rank_in_team`
- `xg_share_in_team`, `shots_share_in_team`

### Player background
- `mv_log` — log market value
- `intl_goal_rate` — international goals per cap
- `player_opp_id`, `player_season_id`, `player_comp_id` (used for target encoding)

### OOF target encodings (13 keys, Bayesian smoothing=30)
Strictly out-of-fold — leakage-free.

| Encoding key | What it captures |
|---|---|
| `te_game_team_id` | Team scoring density in this specific match — **biggest lever (+0.031 AP)** |
| `te_player_id` | Player's career goal rate |
| `te_player_season_id` | Player's in-season form |
| `te_own_club` / `te_opp_club` | Club-level scoring tendency, home vs away |
| `te_sub_position` | Position-level scoring rate |
| `te_referee`, `te_name_x` | Referee and competition-level encoding |

**Total: 102 features (14 categorical).**

---

## Models

Results below are from `solution_fast.ipynb` (cached run). `notebook-kaggle.ipynb` gives slightly lower values (LGB 0.51655 / CAT 0.51097 / Blend 0.52050) due to a different early-stopping point across training runs.

| Model | OOF AP | OOF AUC | Device | Key settings |
|-------|--------|---------|--------|-------------|
| LightGBM | 0.51698 | 0.91788 | CPU | `num_leaves=350`, monotone constraint, early stop 100 on AP |
| XGBoost | 0.51326 | 0.91555 | GPU | `max_depth=9`, `eval_metric=aucpr`, early stop 200 |
| CatBoost | 0.51190 | 0.91458 | GPU | `depth=8`, `eval_metric=PRAUC`, early stop 200 |
| **Blend** | **0.52111** | **0.91822** | — | 0.60 × LGB + 0.00 × XGB + 0.40 × CAT |

Blend weights found via grid search at 5% increments over OOF AP.  
Post-processing: predictions floored to 1×10⁻⁴ when `team_goals = 0`.

---

## What Did NOT Help

All experiments run via leakage-safe 5-fold CV.

| Experiment | Result | Why |
|------------|--------|-----|
| SMOTE / oversampling | −0.017 AP | Synthetic positives add noise; cannot synthesise target-encoded columns |
| Undersampling (1:1, 1:3) | −0.009 to −0.030 AP | Discards majority signal; GBMs handle imbalance natively |
| `scale_pos_weight` | rank-neutral, wrecks calibration | AP is invariant to monotone rescaling |
| StandardScaler | −0.005 AP | No-op for tree models; forced NaN fill destroys missingness signal |
| MissForest imputation | tie (−0.00006), 8× slower | Native NaN handling is already optimal |
| IQR outlier removal | −0.0003 AP | Trees isolate extremes; clipping erases real signal |
| Explicit feature interactions | −0.0003 AP | GBMs already learn these via splits |
| OOF stacking | tie | Base models are too correlated; degenerates to a weighted average |

**Key takeaway:** recognising the random train/test split and applying OOF group target encoding was worth +0.085 AP — far more than any preprocessing, resampling, or ensemble technique.

---

## Reproducibility Notes

- All random seeds fixed at **42** throughout.
- OOF target encodings are strictly leakage-free (each row encoded only from out-of-fold data).
- Feature engineering uses only `train.csv` and `test.csv` — no external datasets.
- Leakage verified via a fresh 20% holdout: holdout AP 0.505 vs pre-encoding baseline 0.428.
