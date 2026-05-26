"""
Data loading for the diabetes-friendly recipe recommender.

archive (2).zip  (CGM + meal diary)
  - blood_sugar_data.csv : Date, Time, Blood Sugar Level
  - food_data.csv        : Date, Time, Meal Type, Food Items

archive (3).zip  (Cleaned Indian Food Dataset - Kaggle)
  - Cleaned_Indian_Food_Dataset.csv
    columns: TranslatedRecipeName, Cleaned-Ingredients, TotalTimeInMins,
             Cuisine, TranslatedInstructions, Ingredient-count

archive (5).zip  (Food GI + Diabetes Suitability DB)
  - pred_food.csv : Food Name, Glycemic Index, Suitable for Diabetes, ...

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


def _extract_gi_food_names(archive5_path: str) -> set[str]:
    """
    archive(5).zip pred_food.csv에서 고유 음식명을 추출해 반환.
    """
    with zipfile.ZipFile(archive5_path) as z:
        with z.open("pred_food.csv") as f:
            df = pd.read_csv(f)

    return set(df["Food Name"].str.lower().unique())


def load_indian_recipes(
    archive3_path: str,
    archive2_path: str = None,
    archive5_path: str = None,
    nrows: int = None,
) -> pd.DataFrame:
    """
    Load Cleaned_Indian_Food_Dataset.csv from archive (3).zip.

    Parameters
    ----------
    archive3_path : path to archive (3).zip (Indian recipes)
    archive2_path : optional. archive(2) CGM 음식명 기준 필터링.
    archive5_path : optional. archive(5) GI DB 음식명 기준 필터링 (archive2와 합집합).
                   둘 다 제공 시 archive(2) OR archive(5)에 있는 레시피를 모두 포함.
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

    # 음식명 기반 필터링: archive(2) OR archive(5) 합집합
    food_names: set[str] = set()
    if archive2_path:
        food_names |= _extract_cgm_food_names(archive2_path)
    if archive5_path:
        food_names |= _extract_gi_food_names(archive5_path)

    if food_names:
        name_lower = df["name"].str.lower()
        mask = name_lower.apply(
            lambda rname: any(food in rname or rname in food for food in food_names)
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


