"""
Model evaluation script.

Metrics:
  - Content-based : Precision@K, average diabetes_score of top-K results
  - Collaborative : RMSE on interactions_validation.csv
"""

import zipfile
import numpy as np
import pandas as pd
from src.data_loader import load_recipes, load_interactions
from src.recommender import DiabetesRecipeRecommender


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def evaluate_cf(model: DiabetesRecipeRecommender, archive_path: str):
    """Compute RMSE on validation set."""
    print("\n[CF Evaluation] Loading validation interactions ...")
    with zipfile.ZipFile(archive_path) as z:
        with z.open("interactions_validation.csv") as f:
            val = pd.read_csv(f)

    cf = model.cf_model
    y_true, y_pred = [], []

    for _, row in val.iterrows():
        uid, rid, rating = int(row["user_id"]), int(row["recipe_id"]), float(row["rating"])
        if uid not in cf.user_index or rid not in cf.recipe_index:
            continue
        u_idx = cf.user_index[uid]
        r_idx = cf.recipe_index[rid]
        pred = float(cf.user_factors[u_idx] @ cf.recipe_factors[r_idx])
        y_true.append(rating)
        y_pred.append(pred)

    if not y_true:
        print("  No overlapping users/recipes in validation set.")
        return

    print(f"  Validation samples : {len(y_true):,}")
    print(f"  RMSE               : {rmse(y_true, y_pred):.4f}")


def evaluate_cb(model: DiabetesRecipeRecommender, n_queries: int = 100):
    """
    Evaluate content-based recommendations.
    Randomly sample ingredient sets and measure:
      - avg diabetes_score of top-10
      - avg similarity of top-10
    """
    print("\n[CB Evaluation] Sampling random ingredient queries ...")
    recipes = model.recipes_df.sample(n=n_queries, random_state=42)

    all_scores, all_sims = [], []
    for _, row in recipes.iterrows():
        ings = row["ingredients"][:5]  # use first 5 ingredients as query
        if not ings:
            continue
        results = model.recommend(ings, top_n=10, min_diabetes_score=0)
        if results.empty:
            continue
        all_scores.extend(results["diabetes_score"].tolist())
        all_sims.extend(results["similarity"].tolist())

    print(f"  Queries evaluated  : {n_queries}")
    print(f"  Avg diabetes score : {np.mean(all_scores):.2f} / 100")
    print(f"  Avg similarity     : {np.mean(all_sims):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=r"C:\Users\gudwl\Downloads\archive.zip")
    parser.add_argument("--max-recipes", type=int, default=20_000)
    args = parser.parse_args()

    print("Loading data and training models for evaluation ...")
    recipes = load_recipes(args.archive, nrows=args.max_recipes)
    interactions = load_interactions(args.archive)

    model = DiabetesRecipeRecommender(n_factors=50)
    model.fit(recipes, interactions)

    evaluate_cb(model)
    evaluate_cf(model, args.archive)
