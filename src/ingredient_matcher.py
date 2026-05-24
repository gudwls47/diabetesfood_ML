"""
Ingredient coverage matcher.

Finds recipes the user can actually make with their available ingredients,
rather than finding recipes that are merely similar.

Coverage = (# recipe ingredients the user has) / (# total recipe ingredients)
Matching uses substring overlap so "chicken breast" matches user input "chicken".
"""

import re
import pandas as pd


def _tokenize(text: str) -> set[str]:
    """Lowercase and split into words, strip punctuation."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _ingredient_covered(recipe_ingredient: str, user_tokens: set[str]) -> bool:
    """
    True if any user ingredient word overlaps with recipe ingredient words.
    e.g. user has 'chicken' -> covers recipe ingredient 'chicken breast'.
    """
    recipe_tokens = _tokenize(recipe_ingredient)
    return bool(recipe_tokens & user_tokens)


def compute_coverage(
    recipe_ingredients: list[str],
    user_ingredients: list[str],
) -> float:
    """Return fraction of recipe ingredients covered by user's pantry (0.0–1.0)."""
    if not recipe_ingredients:
        return 0.0
    user_tokens = set()
    for ing in user_ingredients:
        user_tokens |= _tokenize(ing)

    covered = sum(
        1 for ri in recipe_ingredients if _ingredient_covered(ri, user_tokens)
    )
    return covered / len(recipe_ingredients)


def filter_makeable_recipes(
    recipes_df: pd.DataFrame,
    user_ingredients: list[str],
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """
    Return recipes where the user has at least `min_coverage` fraction
    of the required ingredients, sorted by coverage descending.

    Parameters
    ----------
    recipes_df      : DataFrame with 'ingredients' column (list of strings)
    user_ingredients: list of ingredient strings the user has available
    min_coverage    : minimum fraction of recipe ingredients the user must have
                      (default 0.5 = can make if you have half the ingredients)

    Returns
    -------
    Filtered and sorted DataFrame with added 'coverage' column.
    """
    df = recipes_df.copy()
    df["coverage"] = df["ingredients"].apply(
        lambda ings: compute_coverage(ings, user_ingredients)
    )
    df = df[df["coverage"] >= min_coverage]
    df = df.sort_values("coverage", ascending=False).reset_index(drop=True)
    return df
