"""
Nutritional lookup database built from archive (1).zip.

Loads all 5 FOOD-DATA-GROUP*.csv files and provides per-100g
nutritional values for ingredient matching.

Key columns used:
  food, Carbohydrates, Sugars, Dietary Fiber, Protein, Fat, Caloric Value
"""

import zipfile
import re
import pandas as pd


_COLS = ["food", "Caloric Value", "Fat", "Carbohydrates",
         "Sugars", "Protein", "Dietary Fiber"]

_RENAME = {
    "Caloric Value": "calories",
    "Fat": "fat_g",
    "Carbohydrates": "carbs_g",
    "Sugars": "sugar_g",
    "Protein": "protein_g",
    "Dietary Fiber": "fiber_g",
}


def load_nutrition_db(fooddata1_path: str) -> pd.DataFrame:
    """
    Load all GROUP CSVs from archive (1).zip and return unified DataFrame.
    Indexed by lowercased food name for fast lookup.
    """
    frames = []
    with zipfile.ZipFile(fooddata1_path) as z:
        for name in z.namelist():
            if re.match(r"FINAL FOOD DATASET/FOOD-DATA-GROUP\d+\.csv", name):
                with z.open(name) as f:
                    df = pd.read_csv(f, usecols=_COLS)
                    frames.append(df)

    db = pd.concat(frames, ignore_index=True)
    db = db.rename(columns=_RENAME)
    db["food_lower"] = db["food"].str.lower().str.strip()
    db = db.drop_duplicates(subset="food_lower").reset_index(drop=True)
    return db


def lookup_nutrition(
    ingredient: str,
    db: pd.DataFrame,
) -> dict | None:
    """
    Find the closest nutritional entry for an ingredient name.

    Tries exact match first, then substring match.
    Returns dict with nutritional values per 100g, or None if not found.
    """
    key = ingredient.lower().strip()

    # 1. Exact match
    row = db[db["food_lower"] == key]

    # 2. Ingredient is substring of food name (e.g. "chicken" in "chicken breast")
    if row.empty:
        row = db[db["food_lower"].str.contains(key, regex=False, na=False)]

    # 3. Food name is substring of ingredient (e.g. "potato" in "sweet potato wedges")
    if row.empty:
        matches = db["food_lower"].apply(lambda f: f in key)
        row = db[matches]

    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "food": r["food"],
        "calories": float(r["calories"]),
        "carbs_g": float(r["carbs_g"]),
        "sugar_g": float(r["sugar_g"]),
        "fiber_g": float(r["fiber_g"]),
        "protein_g": float(r["protein_g"]),
        "fat_g": float(r["fat_g"]),
    }


def estimate_recipe_nutrition(
    ingredients: list[str],
    db: pd.DataFrame,
    grams_per_ingredient: float = 100.0,
) -> dict:
    """
    Estimate total nutritional content for a recipe by summing ingredient values.

    Since Food.com recipes don't include gram amounts, we use a fixed
    default of 100g per ingredient as an approximation.

    Returns dict with total nutrient values and match_rate (0-1).
    """
    totals = {k: 0.0 for k in
              ["calories", "carbs_g", "sugar_g", "fiber_g", "protein_g", "fat_g"]}
    matched = 0

    for ing in ingredients:
        result = lookup_nutrition(ing, db)
        if result:
            matched += 1
            scale = grams_per_ingredient / 100.0
            for key in totals:
                totals[key] += result[key] * scale

    totals["net_carbs_g"] = max(0.0, totals["carbs_g"] - totals["fiber_g"])
    totals["match_rate"] = matched / len(ingredients) if ingredients else 0.0
    return totals
