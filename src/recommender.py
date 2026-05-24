"""
Unified diabetes-friendly recipe recommender.

Combines:
  1. Content-based filtering  (ingredient similarity)
  2. Diabetes suitability scoring
  3. Collaborative filtering  (personalisation for known users)
"""

import pandas as pd
import numpy as np
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeFilter
from .diabetes_scorer import compute_diabetes_score, get_diabetes_label


class DiabetesRecipeRecommender:
    def __init__(self, n_factors: int = 50):
        self.cb_model = ContentBasedRecommender()
        self.cf_model = CollaborativeFilter(n_factors=n_factors)
        self.recipes_df = None
        self._cf_fitted = False

    def fit(
        self,
        recipes_df: pd.DataFrame,
        interactions_df: pd.DataFrame = None,
    ) -> "DiabetesRecipeRecommender":
        """
        Train both models.

        Parameters
        ----------
        recipes_df      : output of data_loader.load_recipes()
        interactions_df : optional, output of data_loader.load_interactions()
        """
        # 1. Compute diabetes scores
        self.recipes_df = compute_diabetes_score(recipes_df)

        # 2. Fit content-based model
        self.cb_model.fit(self.recipes_df)

        # 3. Fit collaborative filter (optional)
        if interactions_df is not None and len(interactions_df) > 0:
            self.cf_model.fit(interactions_df)
            self._cf_fitted = True

        return self

    def recommend(
        self,
        ingredients: list[str],
        user_id: int = None,
        top_n: int = 10,
        min_diabetes_score: float = 30.0,
        cf_weight: float = 0.3,
    ) -> pd.DataFrame:
        """
        Recommend diabetes-friendly recipes.

        Parameters
        ----------
        ingredients       : list of available ingredient strings
        user_id           : if provided and CF model is fitted, blend scores
        top_n             : number of recipes to return
        min_diabetes_score: filter threshold (0-100)
        cf_weight         : blend weight for CF score (0 = pure content-based)

        Returns
        -------
        DataFrame sorted by final_score descending
        """
        # Step 1: content-based candidates (3x top_n for re-ranking)
        candidates = self.cb_model.recommend(
            ingredients,
            top_n=top_n * 3,
            min_diabetes_score=min_diabetes_score,
        )

        if candidates.empty:
            return candidates

        # Step 2: blend with CF if available
        if self._cf_fitted and user_id is not None:
            try:
                cf_recs = self.cf_model.recommend_for_user(user_id, top_n=top_n * 3)
                cf_map = dict(zip(cf_recs["recipe_id"], cf_recs["predicted_rating"]))

                # Normalise CF scores to [0, 1]
                cf_vals = np.array(list(cf_map.values()))
                cf_min, cf_max = cf_vals.min(), cf_vals.max()
                cf_range = cf_max - cf_min if cf_max != cf_min else 1.0

                def get_cf_score(recipe_name):
                    # Look up by name via recipes_df
                    match = self.recipes_df[self.recipes_df["name"] == recipe_name]
                    if match.empty:
                        return 0.0
                    rid = match.iloc[0]["id"]
                    raw = cf_map.get(rid, cf_min)
                    return (raw - cf_min) / cf_range

                candidates["cf_score"] = candidates["name"].apply(get_cf_score)
            except ValueError:
                candidates["cf_score"] = 0.0
                cf_weight = 0.0
        else:
            candidates["cf_score"] = 0.0
            cf_weight = 0.0

        # Step 3: compute final score
        # = (1 - cf_weight) * similarity + cf_weight * cf_score
        #   weighted by diabetes_score normalised to [0,1]
        diabetes_norm = candidates["diabetes_score"] / 100.0
        content_score = (1 - cf_weight) * candidates["similarity"]
        collab_score = cf_weight * candidates["cf_score"]

        candidates["final_score"] = (
            (content_score + collab_score) * (0.5 + 0.5 * diabetes_norm)
        ).round(4)

        result = (
            candidates.sort_values("final_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        # Add human-readable diabetes label
        result["diabetes_label"] = result["diabetes_score"].apply(get_diabetes_label)

        return result[
            ["name", "final_score", "diabetes_score", "diabetes_label",
             "calories", "sugar_pct", "carbs_pct", "similarity", "ingredients"]
        ]
