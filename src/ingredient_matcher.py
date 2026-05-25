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


def get_missing_ingredients(
    recipe_ingredients: list[str],
    user_ingredients: list[str],
) -> list[str]:
    """Return list of recipe ingredients the user does NOT have."""
    user_tokens = set()
    for ing in user_ingredients:
        user_tokens |= _tokenize(ing)

    return [
        ri for ri in recipe_ingredients
        if not _ingredient_covered(ri, user_tokens)
    ]


# 자동 완화 단계: 결과 없을 때 순서대로 시도
_FALLBACK_THRESHOLDS = [0.5, 0.3, 0.2, 0.1]


def filter_makeable_recipes(
    recipes_df: pd.DataFrame,
    user_ingredients: list[str],
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """
    Return recipes where the user has at least `min_coverage` fraction
    of the required ingredients.

    재료가 적어 결과가 없을 경우 임계값을 자동으로 낮춰 재시도합니다.
    (0.5 → 0.3 → 0.2 → 0.1)

    Returns
    -------
    DataFrame with added columns:
      coverage          : fraction of recipe ingredients the user has (0-1)
      missing_ingredients: list of ingredients the user still needs
      applied_coverage  : actual threshold used (완화된 경우 변경됨)
    """
    df = recipes_df.copy()
    df["coverage"] = df["ingredients"].apply(
        lambda ings: compute_coverage(ings, user_ingredients)
    )

    # 자동 임계값 완화
    thresholds = sorted(
        set([min_coverage] + [t for t in _FALLBACK_THRESHOLDS if t <= min_coverage]),
        reverse=True,
    )
    # min_coverage보다 낮은 fallback도 포함
    all_thresholds = sorted(
        set([min_coverage] + _FALLBACK_THRESHOLDS),
        reverse=True,
    )

    MIN_RESULTS = 5   # 최소 이 개수 이상 나올 때까지 임계값 완화

    applied = min_coverage
    result = pd.DataFrame()
    for threshold in all_thresholds:
        filtered = df[df["coverage"] >= threshold]
        if len(filtered) >= MIN_RESULTS:
            result = filtered
            applied = threshold
            break
        # 마지막 임계값(0.1)까지 왔는데도 부족하면 있는 것만 반환
        if threshold == all_thresholds[-1] and not filtered.empty:
            result = filtered
            applied = threshold

    if result.empty:
        return result

    # 부족한 재료 계산
    result = result.copy()
    result["missing_ingredients"] = result["ingredients"].apply(
        lambda ings: get_missing_ingredients(ings, user_ingredients)
    )
    result["applied_coverage"] = applied
    result = result.sort_values("coverage", ascending=False).reset_index(drop=True)

    # 임계값이 완화된 경우 사용자에게 알림
    if applied < min_coverage:
        print(f"  [안내] 재료 보유율 {int(min_coverage*100)}% 기준으로는 결과가 없어 "
              f"{int(applied*100)}%로 완화했습니다.")

    return result
