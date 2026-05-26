"""
Diabetes-friendly recipe recommender — full pipeline.

Data sources
------------
- archive.zip      : Food.com 레시피 231,637개 (영양성분 포함)
- CGMacros zip     : 45명 CGM + 식사 영양성분 데이터 -> BG rise ML 모델 학습

Flow
----
1. Korean -> English translation   (translator.py)
2. Ingredient coverage filter      (ingredient_matcher.py)
3. Embedded nutrition from recipe  (칼로리·탄수화물·단백질·지방 직접 사용)
4. BG rise prediction              (bg_model.BGRiseModel)
   - Gradient Boosting: carbs, protein, fat, fiber, pre_bg -> BG rise
5. Diabetes suitability label      (bg_model.BGRiseModel.classify)
6. Rank by suitability then coverage
"""

import pandas as pd

from .translator import translate_ingredients
from .ingredient_matcher import filter_makeable_recipes
from .bg_model import BGRiseModel

_SUIT_RANK = {"적합": 0, "부적합": 1}


class DiabetesRecipeRecommender:
    def __init__(self):
        self.bg_model   = BGRiseModel()
        self.recipes_df = None
        self._bg_fitted = False

    def fit(
        self,
        recipes_df: pd.DataFrame,
        cgmacros_path: str = None,
    ) -> "DiabetesRecipeRecommender":
        """
        Parameters
        ----------
        recipes_df    : output of load_foodcom_recipes()
                        (nutrition columns already embedded: calories, carbs_g,
                         fat_g, protein_g, sugar_g, fiber_g, net_carbs_g)
        cgmacros_path : CGMacros_dateshifted365.zip 경로
        """
        self.recipes_df = recipes_df.reset_index(drop=True)

        if cgmacros_path:
            self.bg_model.fit(cgmacros_path=cgmacros_path)
            self._bg_fitted = True
            print(f"      학습 완료: {self.bg_model._n_samples}건 식사 데이터 "
                  f"(평균 BG rise {self.bg_model._mean_bg_rise:.1f} mg/dL)")

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
        min_coverage        : minimum ingredient coverage (0-1)
        meal_type           : 'breakfast' | 'lunch' | 'dinner'
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

        # Step 3: predict BG rise using embedded nutrition columns
        bg_rise_list  = []
        post_bg_list  = []
        label_list    = []
        desc_list     = []

        for _, row in candidates.iterrows():
            nutr = {
                "carbs_g":     float(row.get("carbs_g",     0) or 0),
                "protein_g":   float(row.get("protein_g",   0) or 0),
                "fat_g":       float(row.get("fat_g",       0) or 0),
                "fiber_g":     float(row.get("fiber_g",     0) or 0),
                "net_carbs_g": float(row.get("net_carbs_g", 0) or 0),
                "calories":    float(row.get("calories",    0) or 0),
            }

            if self._bg_fitted:
                bg_rise = self.bg_model.predict_from_nutrition(
                    nutr, meal_type=meal_type, pre_bg=pre_bg
                )
                label, desc, post_bg = self.bg_model.classify(bg_rise, pre_bg)
            else:
                bg_rise = None
                post_bg = None
                label, desc = "정보 없음", "BG 모델 미학습"

            bg_rise_list.append(bg_rise)
            post_bg_list.append(post_bg)
            label_list.append(label)
            desc_list.append(desc)

        candidates = candidates.reset_index(drop=True)
        candidates["bg_rise_mg_dl"]    = bg_rise_list
        candidates["post_meal_bg"]     = post_bg_list
        candidates["suitability"]      = label_list
        candidates["suitability_desc"] = desc_list

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
