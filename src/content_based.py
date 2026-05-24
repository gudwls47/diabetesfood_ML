"""
Content-based recipe recommender using TF-IDF on ingredient text.

Usage:
    model = ContentBasedRecommender()
    model.fit(recipes_df)
    results = model.recommend(["chicken", "broccoli", "garlic"], top_n=10)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=10_000,
        )
        self.tfidf_matrix = None
        self.recipes_df = None

    def fit(self, recipes_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build TF-IDF matrix from recipe ingredients.
        recipes_df must have 'ingredients_str' and 'diabetes_score' columns.
        """
        self.recipes_df = recipes_df.reset_index(drop=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.recipes_df["ingredients_str"]
        )
        return self

    def recommend(
        self,
        input_ingredients: list[str],
        top_n: int = 10,
        min_diabetes_score: float = 0.0,
    ) -> pd.DataFrame:
        """
        Given a list of ingredient names, return top_n most similar recipes.

        Parameters
        ----------
        input_ingredients : list of ingredient name strings
        top_n             : number of results to return
        min_diabetes_score: filter out recipes below this threshold (0-100)

        Returns
        -------
        DataFrame with columns: name, diabetes_score, similarity,
                                 calories, ingredients
        """
        query = " ".join(input_ingredients)
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        df = self.recipes_df.copy()
        df["similarity"] = sims

        # Filter by diabetes suitability
        if min_diabetes_score > 0:
            df = df[df["diabetes_score"] >= min_diabetes_score]

        result = (
            df.sort_values("similarity", ascending=False)
            .head(top_n)[
                ["name", "diabetes_score", "similarity",
                 "calories", "sugar_pct", "carbs_pct", "ingredients"]
            ]
            .reset_index(drop=True)
        )
        result["diabetes_label"] = result["diabetes_score"].apply(
            lambda s: _label(s)
        )
        return result

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ContentBasedRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)


def _label(score: float) -> str:
    if score >= 75:
        return "매우 적합"
    elif score >= 50:
        return "적합"
    elif score >= 25:
        return "주의 필요"
    return "비권장"
