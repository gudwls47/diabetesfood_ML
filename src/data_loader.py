"""
Data loading and preprocessing for diabetes-friendly recipe recommender.

archive.zip  (Food.com - Western)
  - RAW_recipes.csv         : name, id, ingredients, nutrition, tags, steps
  - interactions_train.csv  : user_id, recipe_id, rating

archive (3).zip  (Cleaned Indian Food Dataset - Kaggle)
  - Cleaned_Indian_Food_Dataset.csv
    columns: TranslatedRecipeName, Cleaned-Ingredients, TotalTimeInMins,
             Cuisine, TranslatedInstructions, Ingredient-count

nutrition column format (list of 7 floats):
  [calories, total_fat_%DV, sugar_%DV, sodium_%DV,
   protein_%DV, sat_fat_%DV, carbohydrates_%DV]
"""

import zipfile
import ast
import pandas as pd
import numpy as np


NUTRITION_COLS = [
    "calories", "total_fat_pct", "sugar_pct",
    "sodium_pct", "protein_pct", "sat_fat_pct", "carbs_pct",
]


def load_recipes(archive_path: str, nrows: int = None) -> pd.DataFrame:
    """Load and clean RAW_recipes.csv from archive zip."""
    with zipfile.ZipFile(archive_path) as z:
        with z.open("RAW_recipes.csv") as f:
            df = pd.read_csv(f, nrows=nrows)

    # Parse nutrition list string -> separate columns
    def parse_nutrition(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [np.nan] * 7

    nutrition = df["nutrition"].apply(parse_nutrition).apply(pd.Series)
    nutrition.columns = NUTRITION_COLS
    df = pd.concat([df.drop(columns=["nutrition"]), nutrition], axis=1)

    # Parse ingredients string -> Python list
    df["ingredients"] = df["ingredients"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df["ingredients_str"] = df["ingredients"].apply(lambda lst: " ".join(lst))

    # Basic cleaning
    df = df.dropna(subset=["name", "ingredients_str", "calories"])
    df = df[df["calories"] > 0].reset_index(drop=True)

    return df


def load_interactions(archive_path: str) -> pd.DataFrame:
    """Load interactions_train.csv (user ratings)."""
    with zipfile.ZipFile(archive_path) as z:
        with z.open("interactions_train.csv") as f:
            df = pd.read_csv(f)
    df = df[["user_id", "recipe_id", "rating"]].dropna()
    df["rating"] = df["rating"].astype(float)
    return df


def load_indian_recipes(archive3_path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load Cleaned_Indian_Food_Dataset.csv from archive (3).zip.

    Returns DataFrame with same key columns as load_recipes():
      name, ingredients (list), ingredients_str, cuisine, cook_time_mins
    so it can be used as a drop-in replacement for the Food.com dataset.

    Source: https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset
    """
    with zipfile.ZipFile(archive3_path) as z:
        with z.open("Cleaned_Indian_Food_Dataset.csv") as f:
            df = pd.read_csv(f, nrows=nrows)

    # Rename to unified column names
    df = df.rename(columns={
        "TranslatedRecipeName": "name",
        "Cleaned-Ingredients": "ingredients_str",
        "TotalTimeInMins": "cook_time_mins",
        "Cuisine": "cuisine",
    })

    # Parse comma-separated ingredient string -> Python list
    df["ingredients"] = df["ingredients_str"].apply(
        lambda x: [i.strip() for i in str(x).split(",") if i.strip()]
    )

    # Add placeholder nutrition columns so recommender doesn't break
    # (actual nutrition is looked up from archive(1).zip per recipe)
    for col in ["calories", "sugar_pct", "carbs_pct", "protein_pct"]:
        df[col] = np.nan

    # Add sequential id
    df = df.reset_index(drop=True)
    df["id"] = df.index

    df = df.dropna(subset=["name", "ingredients_str"]).reset_index(drop=True)
    return df[[
        "id", "name", "ingredients", "ingredients_str",
        "cuisine", "cook_time_mins",
        "calories", "sugar_pct", "carbs_pct", "protein_pct",
    ]]


def load_food_nutrients(fooddata_path: str) -> pd.DataFrame:
    """
    Load USDA FoodData Central and return per-food nutrient pivot.
    Returns DataFrame indexed by food description with key nutrient columns.
    """
    key_nutrients = {
        1003: "protein_g",
        1004: "fat_g",
        1005: "carbs_g",
        1008: "energy_kcal",
        2000: "total_sugars_g",
        1079: "fiber_g",
    }

    with zipfile.ZipFile(fooddata_path) as z:
        prefix = "FoodData_Central_foundation_food_csv_2026-04-30/"

        with z.open(prefix + "food.csv") as f:
            food_df = pd.read_csv(f, usecols=["fdc_id", "description"])

        with z.open(prefix + "food_nutrient.csv") as f:
            nutrient_df = pd.read_csv(
                f, usecols=["fdc_id", "nutrient_id", "amount"]
            )

    nutrient_df = nutrient_df[nutrient_df["nutrient_id"].isin(key_nutrients)]
    nutrient_df["nutrient_name"] = nutrient_df["nutrient_id"].map(key_nutrients)

    pivot = (
        nutrient_df.groupby(["fdc_id", "nutrient_name"])["amount"]
        .mean()
        .unstack("nutrient_name")
        .reset_index()
    )
    merged = food_df.merge(pivot, on="fdc_id", how="inner")
    merged["description_lower"] = merged["description"].str.lower()
    return merged
