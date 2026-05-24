"""
Collaborative filtering using Truncated SVD (Matrix Factorization).

Learns latent user/recipe factors from rating history.
Used to personalise recommendations for returning users.

Usage:
    model = CollaborativeFilter(n_factors=50)
    model.fit(interactions_df)
    recs = model.recommend_for_user(user_id=12345, top_n=10)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import pickle


class CollaborativeFilter:
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)

        self.user_factors = None
        self.recipe_factors = None
        self.user_index = {}      # user_id  -> row idx
        self.recipe_index = {}    # recipe_id -> col idx
        self.idx_to_recipe = {}   # col idx  -> recipe_id
        self.rating_matrix = None

    def fit(self, interactions_df: pd.DataFrame) -> "CollaborativeFilter":
        """
        Build user-recipe rating matrix and decompose with SVD.

        interactions_df must have columns: user_id, recipe_id, rating
        """
        df = interactions_df.copy()

        users = df["user_id"].unique()
        recipes = df["recipe_id"].unique()

        self.user_index = {u: i for i, u in enumerate(users)}
        self.recipe_index = {r: i for i, r in enumerate(recipes)}
        self.idx_to_recipe = {i: r for r, i in self.recipe_index.items()}

        rows = df["user_id"].map(self.user_index)
        cols = df["recipe_id"].map(self.recipe_index)
        data = df["rating"].values

        self.rating_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(recipes)),
        )

        # Decompose: R ≈ U * Sigma * Vt
        self.user_factors = self.svd.fit_transform(self.rating_matrix)
        self.recipe_factors = self.svd.components_.T  # shape: (n_recipes, n_factors)

        return self

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """
        Predict ratings for all unseen recipes and return top_n.

        Returns DataFrame with columns: recipe_id, predicted_rating
        """
        if user_id not in self.user_index:
            raise ValueError(f"user_id {user_id} not found in training data.")

        u_idx = self.user_index[user_id]
        user_vec = self.user_factors[u_idx]  # (n_factors,)
        scores = self.recipe_factors @ user_vec  # (n_recipes,)

        if exclude_seen:
            seen_cols = self.rating_matrix[u_idx].nonzero()[1]
            scores[seen_cols] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_n]

        return pd.DataFrame({
            "recipe_id": [self.idx_to_recipe[i] for i in top_indices],
            "predicted_rating": scores[top_indices].round(3),
        })

    def similar_recipes(self, recipe_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return recipes with most similar latent factor vectors."""
        if recipe_id not in self.recipe_index:
            raise ValueError(f"recipe_id {recipe_id} not found.")

        r_idx = self.recipe_index[recipe_id]
        target_vec = self.recipe_factors[r_idx]

        norms = np.linalg.norm(self.recipe_factors, axis=1)
        target_norm = np.linalg.norm(target_vec)
        sims = (self.recipe_factors @ target_vec) / (norms * target_norm + 1e-9)
        sims[r_idx] = -1  # exclude self

        top_indices = np.argsort(sims)[::-1][:top_n]

        return pd.DataFrame({
            "recipe_id": [self.idx_to_recipe[i] for i in top_indices],
            "similarity": sims[top_indices].round(4),
        })

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "CollaborativeFilter":
        with open(path, "rb") as f:
            return pickle.load(f)
