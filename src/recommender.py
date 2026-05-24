"""
Diabetes-friendly recipe recommender — full pipeline.

Data sources
------------
- archive.zip          : Food.com recipes + user ratings
- archive (1).zip      : Per-food nutritional values (carbs, fiber, sugar …)
- archive (2).zip      : Real CGM + meal diary -> trains BG rise regressor

Flow
----
1. Korean -> English translation  (translator.py)
2. Ingredient coverage filter     (ingredient_matcher.py)
3. Per-recipe nutrition estimate  (nutrition_db.py)
4. BG rise prediction             (bg_model.py)
5. Diabetes suitability label     (bg_model.BGRiseModel.classify)
6. Collaborative filter blend     (collaborative.py)  [optional]
7. Rank and return
"""

import pandas as pd
import numpy as np

from .data_loader import load_recipes, load_interactions
from .translator import translate_ingredients
from .ingredient_matcher import filter_makeable_recipes
from .nutrition_db import load_nutrition_db, estimate_recipe_nutrition
from .bg_model import BGRiseModel
from .collaborative import CollaborativeFilter

_SUIT_RANK = {"적합": 0, "부적합": 1}


class DiabetesRecipeRecommender:
    def __init__(self, n_factors: int = 50):
        self.cf_model = CollaborativeFilter(n_factors=n_factors)
        self.bg_model = BGRiseModel()
        self.recipes_df = None
        self.nutrition_db = None
        self._cf_fitted = False
        self._bg_fitted = False

    def fit(
        self,
        recipes_df: pd.DataFrame,
        nutrition_db: pd.DataFrame,
        archive2_path: str = None,
        interactions_df: pd.DataFrame = None,
    ) -> "DiabetesRecipeRecommender":
        self.recipes_df = recipes_df.reset_index(drop=True)
        self.nutrition_db = nutrition_db

        # Train BG model
        if archive2_path:
            try:
                self.bg_model.fit(archive2_path, nutrition_db)
                self._bg_fitted = True
                print(f"      BG 모델 학습 완료 "
                      f"(샘플 {self.bg_model._n_samples}건, "
                      f"CV RMSE {self.bg_model.cv_rmse:.2f} mg/dL)")
            except Exception as e:
                print(f"      [경고] BG 모델 학습 실패: {e}")

        # Train collaborative filter
        if interactions_df is not None and len(interactions_df) > 0:
            self.cf_model.fit(interactions_df)
            self._cf_fitted = True

        return self

    def recommend(
        self,
        ingredients: list[str],
        user_id: int = None,
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
        user_id             : optional, for CF personalisation
        top_n               : number of results
        min_coverage        : minimum ingredient coverage (0–1)
        meal_type           : 'breakfast'|'lunch'|'dinner'|'snacks'
        pre_bg              : assumed pre-meal blood glucose (mg/dL)
        exclude_suitability : e.g. ['비권장'] to hide unsuitable recipes
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

        # Step 5: CF blend
        if self._cf_fitted and user_id is not None:
            try:
                cf_recs = self.cf_model.recommend_for_user(user_id, top_n=len(candidates))
                cf_map = dict(zip(cf_recs["recipe_id"], cf_recs["predicted_rating"]))
                cf_vals = np.array(list(cf_map.values()))
                cf_range = cf_vals.max() - cf_vals.min() or 1.0
                cf_min = cf_vals.min()
                candidates["cf_score"] = candidates["id"].apply(
                    lambda rid: (cf_map.get(rid, cf_min) - cf_min) / cf_range
                )
            except ValueError:
                candidates["cf_score"] = 0.0
        else:
            candidates["cf_score"] = 0.0

        # Step 6: rank
        candidates["_suit_rank"] = candidates["suitability"].map(
            _SUIT_RANK
        ).fillna(4)
        candidates["_rank_score"] = (
            candidates["coverage"] * 0.7 + candidates["cf_score"] * 0.3
        )
        result = (
            candidates
            .sort_values(["_suit_rank", "_rank_score"], ascending=[True, False])
            .head(top_n)
            .reset_index(drop=True)
        )

        return result[[
            "name", "coverage",
            "carbs_g", "sugar_g", "fiber_g", "net_carbs_g",
            "bg_rise_mg_dl", "post_meal_bg",
            "suitability", "suitability_desc",
            "calories", "ingredients",
        ]]
