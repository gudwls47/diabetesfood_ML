"""
Diabetes suitability scorer.

Scores each recipe 0-100 (higher = more diabetes-friendly).
Uses sugar_%DV and carbohydrates_%DV from the Food.com nutrition column.

Scoring rationale
-----------------
ADA Standards of Care 2025/2026 (diabetesjournals.org/care) does NOT prescribe
fixed per-meal %DV thresholds. Instead it states:
  (1) Reducing overall carbohydrate intake shows the strongest evidence for
      improving glycaemia.
  (2) Minimise added sugars and refined grains.
  (3) Target >= 14 g dietary fibre per 1,000 kcal.
  (4) Nutrition plans must be individualised.

Because the Food.com dataset stores nutrition as FDA %DV values (not grams),
and fibre data is unavailable at the recipe level, we operationalise the ADA
direction using the following empirical thresholds derived from the FDA %DV
reference amounts (added sugars DV = 50 g; total carbs DV = 275 g):

  sugar_pct  : low <10 %DV (~5 g), high >=50 %DV (~25 g)  -- minimise
  carbs_pct  : low <16 %DV (~45 g/meal), high >=60 %DV    -- reduce
  protein_pct: >=10 %DV bonus  (supports satiety)
  calories   : <=400 kcal bonus (portion control)

These are project-level approximations, not official ADA cut-offs.
"""

import pandas as pd
import numpy as np


# Empirical %DV thresholds (see module docstring for rationale)
SUGAR_LOW = 10     # below this -> full sugar score
SUGAR_HIGH = 50    # above this -> zero sugar score
CARBS_LOW = 16     # below this -> full carbs score  (~45 g, ADA low-carb target)
CARBS_HIGH = 60    # above this -> zero carbs score
PROTEIN_BONUS_THRESH = 10
CALORIE_BONUS_THRESH = 400


def _sugar_score(sugar_pct: pd.Series) -> pd.Series:
    """Lower sugar -> higher score (0-40 points)."""
    score = np.where(
        sugar_pct <= SUGAR_LOW, 40,
        np.where(
            sugar_pct >= SUGAR_HIGH, 0,
            40 * (1 - (sugar_pct - SUGAR_LOW) / (SUGAR_HIGH - SUGAR_LOW))
        )
    )
    return pd.Series(score, index=sugar_pct.index)


def _carbs_score(carbs_pct: pd.Series) -> pd.Series:
    """Lower carbs -> higher score (0-40 points)."""
    score = np.where(
        carbs_pct <= CARBS_LOW, 40,
        np.where(
            carbs_pct >= CARBS_HIGH, 0,
            40 * (1 - (carbs_pct - CARBS_LOW) / (CARBS_HIGH - CARBS_LOW))
        )
    )
    return pd.Series(score, index=carbs_pct.index)


def _bonus_score(df: pd.DataFrame) -> pd.Series:
    """Protein and calorie bonus (0-20 points)."""
    protein_bonus = (df["protein_pct"] >= PROTEIN_BONUS_THRESH).astype(int) * 10
    calorie_bonus = (df["calories"] <= CALORIE_BONUS_THRESH).astype(int) * 10
    return protein_bonus + calorie_bonus


def compute_diabetes_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `diabetes_score` column (0-100) to recipe DataFrame.
    Input df must have columns: sugar_pct, carbs_pct, protein_pct, calories.
    """
    df = df.copy()

    # Fill missing values with median (conservative)
    for col in ["sugar_pct", "carbs_pct", "protein_pct", "calories"]:
        df[col] = df[col].fillna(df[col].median())

    df["diabetes_score"] = (
        _sugar_score(df["sugar_pct"])
        + _carbs_score(df["carbs_pct"])
        + _bonus_score(df)
    ).clip(0, 100).round(1)

    return df


def get_diabetes_label(score: float) -> str:
    """Human-readable label for a diabetes score."""
    if score >= 75:
        return "매우 적합"
    elif score >= 50:
        return "적합"
    elif score >= 25:
        return "주의 필요"
    else:
        return "비권장"
