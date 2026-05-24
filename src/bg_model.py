"""
Blood glucose rise regression model.

Trained on archive (2).zip which contains real CGM + meal diary data
from a single individual over 76 days (247 meals, 761 BG readings).

Pipeline
--------
1. Parse food_data.csv  : "Idly (120 gm), Coconut Chutney (50 gm)" -> list of (food, grams)
2. Nutritional lookup   : match each food to archive(1).zip database
3. Build meal features  : carbs_g, sugar_g, fiber_g, protein_g, fat_g, calories, net_carbs_g
4. Match BG readings    : find pre-meal BG and max post-meal BG (30 min – 2 hr window)
5. Train regressor      : features -> BG rise (mg/dL)

Model: RandomForestRegressor (handles small/nonlinear data better than linear for this size)

Limitation: trained on one person's data (Indian cuisine, ~150 samples).
            Predictions are indicative only, not clinical.
"""

import re
import zipfile
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .nutrition_db import load_nutrition_db, lookup_nutrition


FEATURES = ["carbs_g", "sugar_g", "fiber_g", "protein_g", "fat_g",
            "calories", "net_carbs_g"]

# Meal type encoding
MEAL_TYPE_MAP = {
    "breakfast": 0, "lunch": 1, "dinner": 2, "snacks": 3
}


def _parse_food_items(food_str: str) -> list[tuple[str, float]]:
    """
    Parse 'Idly (120 gm), Tea (150 ml)' -> [('Idly', 120.0), ('Tea', 150.0)]
    Items without a gram value get 100g default.
    """
    items = []
    for part in food_str.split(","):
        part = part.strip()
        match = re.match(r"(.+?)\s*\((\d+(?:\.\d+)?)\s*(?:gm|ml|g)\)", part, re.I)
        if match:
            items.append((match.group(1).strip(), float(match.group(2))))
        elif part:
            items.append((part, 100.0))
    return items


def _meal_nutrition(food_str: str, nutrition_db: pd.DataFrame) -> dict | None:
    """Return aggregated nutritional features for a meal string, or None."""
    items = _parse_food_items(food_str)
    totals = {k: 0.0 for k in FEATURES[:-1]}  # all except net_carbs
    matched = 0

    for food_name, grams in items:
        result = lookup_nutrition(food_name, nutrition_db)
        if result:
            matched += 1
            scale = grams / 100.0
            for key in ["carbs_g", "sugar_g", "fiber_g", "protein_g", "fat_g", "calories"]:
                totals[key] += result[key] * scale

    if matched == 0:
        return None

    totals["net_carbs_g"] = max(0.0, totals["carbs_g"] - totals["fiber_g"])
    return totals


def _build_training_data(
    archive2_path: str,
    nutrition_db: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) training pairs from archive (2).zip.
    X: meal nutritional features
    y: post-meal blood glucose rise (mg/dL)
    """
    with zipfile.ZipFile(archive2_path) as z:
        with z.open("blood_sugar_data.csv") as f:
            bg = pd.read_csv(f)
        with z.open("food_data.csv") as f:
            food = pd.read_csv(f)

    bg["datetime"] = pd.to_datetime(
        bg["Date"] + " " + bg["Time"], format="%d-%m-%Y %I:%M %p"
    )
    food["datetime"] = pd.to_datetime(
        food["Date"] + " " + food["Time"], format="%d-%m-%Y %I:%M %p"
    )
    bg = bg.sort_values("datetime").reset_index(drop=True)
    food["Meal Type"] = food["Meal Type"].str.strip().str.lower()

    rows = []
    for _, meal in food.iterrows():
        meal_time = meal["datetime"]

        # Pre-meal BG: last reading before meal
        pre = bg[bg["datetime"] <= meal_time]["Blood Sugar Level"]
        if pre.empty:
            continue
        pre_bg = float(pre.iloc[-1])

        # Post-meal BG: max reading 30 min – 2 hr after meal
        post_mask = (
            (bg["datetime"] >= meal_time + pd.Timedelta(minutes=30)) &
            (bg["datetime"] <= meal_time + pd.Timedelta(hours=2))
        )
        post = bg[post_mask]["Blood Sugar Level"]
        if post.empty:
            continue
        post_bg = float(post.max())

        bg_rise = post_bg - pre_bg
        if bg_rise < 0:
            continue  # drop noise / overlapping meals

        nutrition = _meal_nutrition(meal["Food Items"], nutrition_db)
        if nutrition is None:
            continue

        nutrition["meal_type"] = MEAL_TYPE_MAP.get(meal["Meal Type"], 3)
        nutrition["pre_bg"] = pre_bg
        nutrition["bg_rise"] = bg_rise
        rows.append(nutrition)

    df = pd.DataFrame(rows)
    X = df[FEATURES + ["meal_type", "pre_bg"]]
    y = df["bg_rise"]
    return X, y


class BGRiseModel:
    """Predicts post-meal blood glucose rise (mg/dL) from meal nutrition features."""

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(
                n_estimators=200, max_depth=4,
                random_state=42, min_samples_leaf=3,
            )),
        ])
        self._trained = False
        self._cv_rmse = None

    def fit(
        self,
        archive2_path: str,
        nutrition_db: pd.DataFrame,
    ) -> "BGRiseModel":
        X, y = _build_training_data(archive2_path, nutrition_db)

        if len(X) < 10:
            raise ValueError(
                f"Too few training samples ({len(X)}). "
                "Check archive (2).zip path and nutrition DB."
            )

        self.model.fit(X, y)
        self._trained = True

        # Cross-validation RMSE for reporting
        scores = cross_val_score(
            self.model, X, y,
            scoring="neg_root_mean_squared_error", cv=min(5, len(X) // 5)
        )
        self._cv_rmse = float(-scores.mean())
        self._n_samples = len(X)
        return self

    def predict(
        self,
        nutrition: dict,
        meal_type: str = "lunch",
        pre_bg: float = 110.0,
    ) -> float:
        """
        Predict blood glucose rise (mg/dL) for a meal.

        Parameters
        ----------
        nutrition : dict with keys matching FEATURES
        meal_type : 'breakfast' | 'lunch' | 'dinner' | 'snacks'
        pre_bg    : assumed fasting/pre-meal blood glucose (mg/dL)

        Returns
        -------
        Predicted BG rise in mg/dL (rounded to 1 decimal)
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        row = {k: nutrition.get(k, 0.0) for k in FEATURES}
        row["meal_type"] = MEAL_TYPE_MAP.get(meal_type.lower(), 3)
        row["pre_bg"] = pre_bg

        X = pd.DataFrame([row])
        pred = float(self.model.predict(X)[0])
        return round(max(0.0, pred), 1)

    def classify(self, bg_rise: float) -> tuple[str, str]:
        """
        Classify diabetes suitability from predicted BG rise.

        Thresholds based on ADA post-meal target < 180 mg/dL
        assuming typical T2D pre-meal baseline ~130 mg/dL
        (safe rise budget = 50 mg/dL).
        """
        if bg_rise <= 20:
            return "적합", f"혈당 상승 예측 +{bg_rise:.0f} mg/dL (매우 낮음)"
        elif bg_rise <= 40:
            return "적합", f"혈당 상승 예측 +{bg_rise:.0f} mg/dL (낮음)"
        elif bg_rise <= 60:
            return "주의", f"혈당 상승 예측 +{bg_rise:.0f} mg/dL (보통)"
        elif bg_rise <= 80:
            return "고주의", f"혈당 상승 예측 +{bg_rise:.0f} mg/dL (높음)"
        else:
            return "비권장", f"혈당 상승 예측 +{bg_rise:.0f} mg/dL (매우 높음)"

    @property
    def cv_rmse(self) -> float | None:
        return self._cv_rmse

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BGRiseModel":
        with open(path, "rb") as f:
            return pickle.load(f)
