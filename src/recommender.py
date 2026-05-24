"""
Diabetes-friendly recipe recommender — unified pipeline.

Flow
----
1. Ingredient matching  : filter recipes the user can actually make
                          (coverage >= min_coverage)
2. Blood glucose calc   : estimate carbs (g), BG rise (mg/dL) per recipe
3. Suitability judgment : classify as 적합 / 주의 / 고주의 / 비권장
4. Personalisation      : optionally blend collaborative filter scores
5. Rank & return        : sort by suitability then coverage
"""

import pandas as pd
import numpy as np

from .data_loader import load_recipes, load_interactions
from .ingredient_matcher import filter_makeable_recipes
from .blood_glucose import compute_blood_glucose_info
from .collaborative import CollaborativeFilter


# Suitability rank for sorting (lower = better)
_SUIT_RANK = {"적합": 0, "주의": 1, "고주의": 2, "비권장": 3}


class DiabetesRecipeRecommender:
    def __init__(self, n_factors: int = 50):
        self.cf_model = CollaborativeFilter(n_factors=n_factors)
        self.recipes_df = None
        self._cf_fitted = False

    def fit(
        self,
        recipes_df: pd.DataFrame,
        interactions_df: pd.DataFrame = None,
    ) -> "DiabetesRecipeRecommender":
        """
        Prepare recipe data and optionally train collaborative filter.

        Parameters
        ----------
        recipes_df      : output of data_loader.load_recipes()
        interactions_df : optional, output of data_loader.load_interactions()
        """
        self.recipes_df = compute_blood_glucose_info(recipes_df)

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
        exclude_suitability: list[str] = None,
    ) -> pd.DataFrame:
        """
        Recommend diabetes-friendly recipes the user can make.

        Parameters
        ----------
        ingredients          : list of ingredients the user has
        user_id              : if provided, blend CF personalisation
        top_n                : number of recipes to return
        min_coverage         : min fraction of recipe ingredients user must have
                               (0.5 = must have at least half the ingredients)
        exclude_suitability  : list of labels to exclude, e.g. ['비권장']

        Returns
        -------
        DataFrame with columns:
          name, coverage, carbs_g, sugar_g, bg_rise_mg_dl,
          suitability, suitability_desc, calories
        """
        # Step 1: filter to makeable recipes
        candidates = filter_makeable_recipes(
            self.recipes_df, ingredients, min_coverage=min_coverage
        )

        if candidates.empty:
            return candidates

        # Step 2: optionally exclude unsuitable categories
        if exclude_suitability:
            candidates = candidates[
                ~candidates["suitability"].isin(exclude_suitability)
            ]

        if candidates.empty:
            return candidates

        # Step 3: blend CF if available
        if self._cf_fitted and user_id is not None:
            try:
                cf_recs = self.cf_model.recommend_for_user(
                    user_id, top_n=len(candidates)
                )
                cf_map = dict(zip(cf_recs["recipe_id"], cf_recs["predicted_rating"]))
                cf_vals = np.array(list(cf_map.values()))
                cf_min = cf_vals.min()
                cf_range = cf_vals.max() - cf_min or 1.0

                def cf_score(recipe_id):
                    raw = cf_map.get(recipe_id, cf_min)
                    return (raw - cf_min) / cf_range

                candidates["cf_score"] = candidates["id"].apply(cf_score)
            except ValueError:
                candidates["cf_score"] = 0.0
        else:
            candidates["cf_score"] = 0.0

        # Step 4: rank — primary: suitability, secondary: coverage + cf_score
        candidates["_suit_rank"] = candidates["suitability"].map(_SUIT_RANK)
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
            "carbs_g", "sugar_g", "bg_rise_mg_dl",
            "suitability", "suitability_desc",
            "calories", "ingredients",
        ]]
