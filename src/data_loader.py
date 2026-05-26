"""
Data loading for the diabetes-friendly recipe recommender.

archive (2).zip  (CGM + meal diary)
  - blood_sugar_data.csv : Date, Time, Blood Sugar Level
  - food_data.csv        : Date, Time, Meal Type, Food Items

archive (3).zip  (Cleaned Indian Food Dataset - Kaggle)
  - Cleaned_Indian_Food_Dataset.csv
    columns: TranslatedRecipeName, Cleaned-Ingredients, TotalTimeInMins,
             Cuisine, TranslatedInstructions, Ingredient-count

Source: https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset
"""

import re
import zipfile
import pandas as pd
import numpy as np


def _extract_cgm_food_names(archive2_path: str) -> set[str]:
    """
    archive(2).zip food_data.csv에서 고유 음식명을 추출해 반환.
    예) "Idly (120 gm), Sambar (100 gm)" -> {'idly', 'sambar'}
    """
    with zipfile.ZipFile(archive2_path) as z:
        with z.open("food_data.csv") as f:
            food_df = pd.read_csv(f)

    names = set()
    for items in food_df["Food Items"]:
        for part in str(items).split(","):
            name = re.sub(r"\(.*?\)", "", part).strip().lower()
            if name:
                names.add(name)
    return names


def load_indian_recipes(
    archive3_path: str,
    archive2_path: str = None,
    nrows: int = None,
) -> pd.DataFrame:
    """
    Load Cleaned_Indian_Food_Dataset.csv from archive (3).zip.

    Parameters
    ----------
    archive3_path : path to archive (3).zip (Indian recipes)
    archive2_path : optional. If provided, only recipes whose name
                   matches foods in archive(2).zip CGM data are returned.
                   이렇게 하면 실제 혈당 측정 데이터가 있는 음식만 추천됩니다.
    nrows         : limit rows (for testing)

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

    # CGM 음식 기반 필터링
    if archive2_path:
        cgm_foods = _extract_cgm_food_names(archive2_path)
        name_lower = df["name"].str.lower()
        mask = name_lower.apply(
            lambda rname: any(food in rname or rname in food for food in cgm_foods)
        )
        df = df[mask].reset_index(drop=True)

    # Parse comma-separated ingredient string -> Python list
    df["ingredients"] = df["ingredients_str"].apply(
        lambda x: [i.strip() for i in str(x).split(",") if i.strip()]
    )

    # Add placeholder nutrition columns
    for col in ["calories", "sugar_pct", "carbs_pct", "protein_pct"]:
        df[col] = np.nan

    df = df.reset_index(drop=True)
    df["id"] = df.index

    df = df.dropna(subset=["name", "ingredients_str"]).reset_index(drop=True)
    return df[[
        "id", "name", "ingredients", "ingredients_str",
        "cuisine", "cook_time_mins",
        "calories", "sugar_pct", "carbs_pct", "protein_pct",
    ]]


