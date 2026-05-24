"""
Blood glucose impact estimator.

Estimates how much a recipe will raise blood glucose (mg/dL) and
whether the result is suitable for people with diabetes.

Method
------
The Food.com nutrition column stores values as FDA %DV:
  carbs_pct : % of daily value for total carbohydrates (FDA DV = 275 g)
  sugar_pct : % of daily value for added sugars        (FDA DV = 50 g)

Step 1 – Convert %DV to grams:
  carbs_g = carbs_pct / 100 * 275
  sugar_g = sugar_pct / 100 * 50

Step 2 – Estimate post-meal blood glucose rise (ΔBG):
  ΔBG (mg/dL) ≈ carbs_g × ISF
  where ISF (insulin sensitivity factor) = 3 mg/dL per gram of carbohydrate.
  This is a commonly cited ballpark for educational use in T2D carb-counting
  (actual ISF varies significantly by individual, medication, and activity level).

Step 3 – Classify suitability:
  ADA Standards of Care 2025 targets post-meal BG < 180 mg/dL.
  Assuming a typical T2D fasting baseline of ~130 mg/dL:
    safe_rise = 180 - 130 = 50 mg/dL  → carbs < ~17 g  (적합)
  We use more practical ADA meal carb guidelines for classification:
    < 30 g carbs  → 적합    (low-carb meal)
    30–45 g       → 주의    (moderate; ADA lower bound for T2D)
    45–60 g       → 고주의  (ADA upper bound for T2D)
    > 60 g        → 비권장  (exceeds ADA per-meal guideline)

References:
  ADA Standards of Care 2025, Section 5
  https://diabetesjournals.org/care/article/48/Supplement_1/S86/157563/

Disclaimer: This is a simplified educational estimate, not a clinical tool.
"""

import pandas as pd
import numpy as np

# FDA daily value reference amounts
CARBS_DV_G = 275.0   # grams per day
SUGAR_DV_G = 50.0    # grams per day (added sugars)

# Insulin sensitivity factor: mg/dL rise per gram of carbohydrate
ISF = 3.0

# ADA per-meal carbohydrate thresholds (grams)
CARBS_LOW_G = 30.0    # below this -> 적합
CARBS_MED_G = 45.0    # below this -> 주의
CARBS_HIGH_G = 60.0   # below this -> 고주의 ; above -> 비권장


def estimate_blood_glucose_rise(carbs_pct: float) -> float:
    """
    Estimate post-meal blood glucose rise in mg/dL from carbs_%DV.
    Returns float (mg/dL).
    """
    carbs_g = carbs_pct / 100.0 * CARBS_DV_G
    return round(carbs_g * ISF, 1)


def get_carbs_g(carbs_pct: float) -> float:
    """Convert carbs %DV to grams."""
    return round(carbs_pct / 100.0 * CARBS_DV_G, 1)


def get_sugar_g(sugar_pct: float) -> float:
    """Convert sugar %DV to grams."""
    return round(sugar_pct / 100.0 * SUGAR_DV_G, 1)


def classify_suitability(carbs_g: float) -> tuple[str, str]:
    """
    Classify diabetes suitability based on estimated carbs in grams.

    Returns
    -------
    (label, description) tuple
      label       : '적합' | '주의' | '고주의' | '비권장'
      description : short explanation
    """
    if carbs_g < CARBS_LOW_G:
        return "적합", f"탄수화물 {carbs_g:.1f}g (30g 미만, 저탄수화물 식단)"
    elif carbs_g < CARBS_MED_G:
        return "주의", f"탄수화물 {carbs_g:.1f}g (30~45g, ADA 권장 하한선 수준)"
    elif carbs_g < CARBS_HIGH_G:
        return "고주의", f"탄수화물 {carbs_g:.1f}g (45~60g, ADA 권장 상한선 근접)"
    else:
        return "비권장", f"탄수화물 {carbs_g:.1f}g (60g 초과, ADA 1회 식사 권장량 초과)"


def compute_blood_glucose_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add blood glucose estimate columns to recipe DataFrame.

    Added columns
    -------------
    carbs_g         : estimated carbohydrates in grams
    sugar_g         : estimated added sugars in grams
    bg_rise_mg_dl   : estimated post-meal blood glucose rise (mg/dL)
    suitability     : '적합' | '주의' | '고주의' | '비권장'
    suitability_desc: short description of why
    """
    df = df.copy()

    for col in ["carbs_pct", "sugar_pct"]:
        df[col] = df[col].fillna(df[col].median())

    df["carbs_g"] = df["carbs_pct"].apply(get_carbs_g)
    df["sugar_g"] = df["sugar_pct"].apply(get_sugar_g)
    df["bg_rise_mg_dl"] = df["carbs_pct"].apply(estimate_blood_glucose_rise)

    suitability_info = df["carbs_g"].apply(classify_suitability)
    df["suitability"] = suitability_info.apply(lambda x: x[0])
    df["suitability_desc"] = suitability_info.apply(lambda x: x[1])

    return df
