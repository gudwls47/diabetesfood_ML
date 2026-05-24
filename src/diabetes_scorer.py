"""
Diabetes suitability scorer.

Scores each recipe 0-100 (higher = more diabetes-friendly).
Uses sugar_%DV and carbohydrates_%DV from nutrition column,
optionally enriched with actual sugar_g from USDA FoodData.

Scoring logic (rule-based thresholds from ADA dietary guidelines):
  - sugar_pct  <=  5 %DV  ->  full sugar score
  - carbs_pct  <= 15 %DV  ->  full carbs score
  - protein_pct >= 10 %DV ->  bonus
  - calories   <= 400      ->  bonus
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# %DV thresholds for diabetes suitability
SUGAR_LOW = 5      # ideal upper limit (%DV)
SUGAR_HIGH = 25    # penalise heavily above this
CARBS_LOW = 15
CARBS_HIGH = 45
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
