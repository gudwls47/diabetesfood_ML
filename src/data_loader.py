"""
Data loading for the diabetes-friendly recipe recommender.

archive (3).zip  (Cleaned Indian Food Dataset - Kaggle)
  - Cleaned_Indian_Food_Dataset.csv
    columns: TranslatedRecipeName, Cleaned-Ingredients, TotalTimeInMins,
             Cuisine, TranslatedInstructions, Ingredient-count

Source: https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset
"""

import zipfile

import numpy as np
import pandas as pd


def load_indian_recipes(
    archive3_path: str,
    nrows: int = None,
) -> pd.DataFrame:
    """
    Load Cleaned_Indian_Food_Dataset.csv from archive (3).zip.

    Parameters
    ----------
    archive3_path : path to archive (3).zip
    nrows         : limit rows (for testing)

    Returns all 5,938 recipes — BG prediction is now handled by the
    CGMacros ML model (nutrition-feature based), so no food-name
    filtering is needed.

    Source: https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset
    """
    with zipfile.ZipFile(archive3_path) as z:
        with z.open("Cleaned_Indian_Food_Dataset.csv") as f:
            df = pd.read_csv(f, nrows=nrows)

    df = df.rename(columns={
        "TranslatedRecipeName": "name",
        "Cleaned-Ingredients":  "ingredients_str",
        "TotalTimeInMins":      "cook_time_mins",
        "Cuisine":              "cuisine",
    })

    df["ingredients"] = df["ingredients_str"].apply(
        lambda x: [i.strip() for i in str(x).split(",") if i.strip()]
    )

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
