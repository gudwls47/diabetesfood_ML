"""
Diabetes-friendly recipe recommender — full pipeline.

Data sources
------------
- archive (1).zip : Per-food nutritional values (carbs, fiber, sugar …)
- archive (2).zip : Real CGM + meal diary -> trains BG rise regressor
- archive (3).zip : Cleaned Indian recipes (5,938개)

Flow
----
1. Korean -> English translation  (translator.py)
2. Ingredient coverage filter     (ingredient_matcher.py)
3. Per-recipe nutrition estimate  (nutrition_db.py)
4. BG rise prediction             (bg_model.py)
5. Diabetes suitability label     (bg_model.BGRiseModel.classify)
6. Rank by coverage and return
"""

import pandas as pd

from .translator import translate_ingredients
from .ingredient_matcher import filter_makeable_recipes
from .nutrition_db import estimate_recipe_nutrition
from .bg_model import BGRiseModel

_SUIT_RANK = {"적합": 0, "부적합": 1}


class DiabetesRecipeRecommender:
    def __init__(self):
        self.bg_model = BGRiseModel()
        self.recipes_df = None
        self.nutrition_db = None
        self._bg_fitted = False

    def fit(
        self,
        recipes_df: pd.DataFrame,
        nutrition_db: pd.DataFrame,
        archive2_path: str = None,
    ) -> "DiabetesRecipeRecommender":
        """
        Parameters
        ----------
        recipes_df    : output of load_indian_recipes()
        nutrition_db  : output of load_nutrition_db()
        archive2_path : (미사용) API 호환 유지용
        """
        self.recipes_df = recipes_df.reset_index(drop=True)
        self.nutrition_db = nutrition_db
        self.bg_model.fit()
        self._bg_fitted = True
        return self

    def recommend(
        self,
        ingredients: list[str],
        top_n: int = 10,
        min_coverage: float = 0.5,
        meal_type: str = "lunch",
        pre_bg: float = 110.0,
        exclude_suitability: list[str] = None,
    ) -> pd.DataFrame:
        """
        Recommend diabetes-friendly recipes.

        Parameters
        ----------
        ingredients         : Korean or English ingredient list
        top_n               : number of results
        min_coverage        : minimum ingredient coverage (0–1)
        meal_type           : 'breakfast'|'lunch'|'dinner'|'snacks'
        pre_bg              : pre-meal blood glucose (mg/dL)
        exclude_suitability : e.g. ['부적합'] to hide unsuitable recipes
        """
        # Step 1: translate Korean -> English
        en_ingredients = translate_ingredients(ingredients)

        # Step 2: filter to makeable recipes
        candidates = filter_makeable_recipes(
            self.recipes_df, en_ingredients, min_coverage=min_coverage
        )
        if candidates.empty:
            return candidates

        # Step 3: estimate nutrition + predict BG for each candidate
        nutrition_rows = []
        for _, row in candidates.iterrows():
            nutr = estimate_recipe_nutrition(row["ingredients"], self.nutrition_db)
            if self._bg_fitted:
                bg_rise = self.bg_model.predict(nutr, meal_type, pre_bg)
                label, desc, post_bg = self.bg_model.classify(bg_rise, pre_bg)
            else:
                bg_rise = None
                post_bg = None
                label, desc = "정보 없음", "BG 모델 미학습"
            nutrition_rows.append({
                **nutr,
                "bg_rise_mg_dl": bg_rise,
                "post_meal_bg": post_bg,
                "suitability": label,
                "suitability_desc": desc,
            })

        nutr_df = pd.DataFrame(nutrition_rows)
        candidates = candidates.reset_index(drop=True)
        candidates = pd.concat([candidates, nutr_df], axis=1)

        # Step 4: exclude unwanted suitability labels
        if exclude_suitability:
            candidates = candidates[
                ~candidates["suitability"].isin(exclude_suitability)
            ]
        if candidates.empty:
            return candidates

        # Step 5: rank by suitability then coverage
        candidates["_suit_rank"] = candidates["suitability"].map(
            _SUIT_RANK
        ).fillna(4)
        result = (
            candidates
            .sort_values(["_suit_rank", "coverage"], ascending=[True, False])
            .head(top_n)
            .reset_index(drop=True)
        )

        return result[[
            "name", "coverage", "missing_ingredients",
            "carbs_g", "sugar_g", "fiber_g", "net_carbs_g",
            "bg_rise_mg_dl", "post_meal_bg",
            "suitability", "suitability_desc",
            "calories", "ingredients",
        ]]
