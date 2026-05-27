"""
Data loading for the diabetes-friendly recipe recommender.

archive.zip  (Food.com Recipes Dataset - Kaggle)
  - RAW_recipes.csv  : 231,637개 레시피
    columns: name, id, minutes, tags, nutrition, ingredients, ...

    nutrition 필드 형식: [calories, fat%DV, sugar%DV, sodium%DV,
                          protein%DV, sat_fat%DV, carbohydrates%DV]
    %DV 기준: carbs 275g, fat 78g, protein 50g, sugar 50g

Source: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
"""

import ast
import zipfile

import numpy as np
import pandas as pd


# %DV -> 실제 g 변환 계수 (2000kcal 기준)
_CARBS_DV_G   = 275.0
_FAT_DV_G     = 78.0
_PROTEIN_DV_G = 50.0
_SUGAR_DV_G   = 50.0

# 당뇨 추천에서 제외할 레시피 태그
_EXCLUDE_TAGS = frozenset({
    "alcoholic", "cocktails-mocktails", "beer", "spirits",
    "punch-cocktails-mocktails",
})

# 레시피명에 포함 시 제외할 알코올 키워드
_EXCLUDE_NAME_KEYWORDS = frozenset({
    "rum", "vodka", "whiskey", "whisky", "bourbon", "gin", "tequila",
    "beer", "wine", "champagne", "brandy", "liqueur", "kahlua",
    "baileys", "frangelico", "schnapps", "amaretto", "cointreau",
    "martini", "cocktail", "margarita", "mojito", "daiquiri",
    "sangria", "jello shot", "jell-o shot", "shot glass",
})


def _parse_nutrition(s) -> list | None:
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list) and len(val) >= 7:
            return val
    except Exception:
        pass
    return None


def _parse_list(s) -> list:
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def load_foodcom_recipes(
    archive_path: str,
    max_calories: float = 3000.0,
    max_carbs_g: float  = 300.0,
    diabetic_only: bool = False,
    nrows: int = None,
    serving_kcal: float = 600.0,
    exclude_inappropriate: bool = True,
) -> pd.DataFrame:
    """
    Load RAW_recipes.csv from archive.zip (Food.com dataset).

    Parameters
    ----------
    archive_path          : archive.zip 경로
    max_calories          : 이상치 제거용 최대 칼로리 (기본 3,000 kcal)
    max_carbs_g           : 이상치 제거용 최대 탄수화물 (기본 300 g)
    diabetic_only         : True 시 'diabetic' 또는 'low-carb' 태그 레시피만 반환
    nrows                 : 로드 행 수 제한 (테스트용)
    serving_kcal          : 1인분 기준 칼로리 (기본 600 kcal).
                            Food.com 영양성분은 레시피 전체(다인분) 기준이므로
                            calories > serving_kcal 인 레시피는 비례 축소해 1인분으로 정규화.
                            0 이하면 정규화 생략.
    exclude_inappropriate : True 시 알코올·칵테일류 레시피 제외 (기본 True)

    Returns
    -------
    DataFrame with columns:
      id, name, ingredients, tags,
      calories, carbs_g, fat_g, protein_g, sugar_g,
      fiber_g (=0, 데이터 없음), net_carbs_g (=carbs_g)
    """
    with zipfile.ZipFile(archive_path) as z:
        with z.open("RAW_recipes.csv") as f:
            df = pd.read_csv(f, nrows=nrows)

    # 영양성분 파싱
    df["_nutr"] = df["nutrition"].apply(_parse_nutrition)
    df = df[df["_nutr"].notna()].copy()

    df["calories"]    = df["_nutr"].apply(lambda x: float(x[0]))
    df["carbs_g"]     = df["_nutr"].apply(lambda x: float(x[6]) * _CARBS_DV_G   / 100)
    df["fat_g"]       = df["_nutr"].apply(lambda x: float(x[1]) * _FAT_DV_G     / 100)
    df["protein_g"]   = df["_nutr"].apply(lambda x: float(x[4]) * _PROTEIN_DV_G / 100)
    df["sugar_g"]     = df["_nutr"].apply(lambda x: float(x[2]) * _SUGAR_DV_G   / 100)
    df["fiber_g"]     = 0.0    # Food.com 데이터에 섬유질 정보 없음

    # 이상치 제거
    df = df[
        (df["calories"] >= 1) &
        (df["calories"] <= max_calories) &
        (df["carbs_g"]  >= 0) &
        (df["carbs_g"]  <= max_carbs_g)
    ].copy()

    # 1인분 기준 정규화
    # Food.com 영양성분은 레시피 전체 기준 — calories > serving_kcal 이면 비례 축소
    if serving_kcal and serving_kcal > 0:
        ratio = np.minimum(1.0, serving_kcal / df["calories"].clip(lower=1.0))
        for col in ("carbs_g", "fat_g", "protein_g", "sugar_g"):
            df[col] = df[col] * ratio
        df["calories"] = df["calories"] * ratio

    df["net_carbs_g"] = df["carbs_g"]  # 섬유질 없으므로 탄수화물 그대로 사용

    # 재료 파싱
    df["ingredients"] = df["ingredients"].apply(_parse_list)
    df["tags"]        = df["tags"].apply(_parse_list)

    # 부적절 레시피 제거 (알코올·칵테일류)
    if exclude_inappropriate:
        # 태그 기반 제거
        tag_mask = df["tags"].apply(
            lambda tags: not bool(set(tags) & _EXCLUDE_TAGS)
        )
        # 레시피명 키워드 기반 제거
        def _name_ok(name: str) -> bool:
            low = str(name).lower()
            return not any(kw in low for kw in _EXCLUDE_NAME_KEYWORDS)

        name_mask = df["name"].apply(_name_ok)
        before = len(df)
        df = df[tag_mask & name_mask].copy()
        removed = before - len(df)
        if removed:
            print(f"      [필터] 부적절 레시피 {removed:,}개 제거 (알코올·칵테일류)")

    # 당뇨/저탄수화물 태그 필터
    if diabetic_only:
        target_tags = {"diabetic", "low-carb", "diabetic-friendly",
                       "low-carbohydrate", "sugar-free"}
        mask = df["tags"].apply(
            lambda tags: bool(set(tags) & target_tags)
        )
        df = df[mask].copy()

    df = df.dropna(subset=["name"]).reset_index(drop=True)
    df["id"] = df.index

    return df[[
        "id", "name", "ingredients", "tags",
        "calories", "carbs_g", "fat_g", "protein_g",
        "sugar_g", "fiber_g", "net_carbs_g",
    ]]
