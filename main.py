"""
Diabetes-Friendly Recipe Recommender — main entry point.

Quick start:
    python main.py

Or import programmatically:
    from src import DiabetesRecipeRecommender
"""

import argparse
import sys
from pathlib import Path

from src.data_loader import load_recipes, load_interactions
from src.recommender import DiabetesRecipeRecommender

# Default paths (adjust to your local setup)
DEFAULT_ARCHIVE = r"C:\Users\gudwl\Downloads\archive.zip"
DEFAULT_FOODDATA = r"C:\Users\gudwl\Downloads\FoodData_Central_foundation_food_csv_2026-04-30.zip"


def build_recommender(
    archive_path: str,
    max_recipes: int = 50_000,
    use_cf: bool = True,
) -> DiabetesRecipeRecommender:
    """Load data and train all models."""
    print(f"[1/3] Loading recipes (max {max_recipes:,}) ...")
    recipes = load_recipes(archive_path, nrows=max_recipes)
    print(f"      -> {len(recipes):,} recipes loaded")

    interactions = None
    if use_cf:
        print("[2/3] Loading user interactions ...")
        interactions = load_interactions(archive_path)
        print(f"      -> {len(interactions):,} ratings loaded")
    else:
        print("[2/3] Skipping collaborative filter (use_cf=False)")

    print("[3/3] Training models ...")
    model = DiabetesRecipeRecommender(n_factors=50)
    model.fit(recipes, interactions)
    print("      -> Done!\n")

    return model


def interactive_session(model: DiabetesRecipeRecommender):
    """Simple CLI loop for interactive recipe queries."""
    print("=" * 60)
    print("  당뇨 환자를 위한 레시피 추천 시스템")
    print("  (종료: 'q' 입력)")
    print("=" * 60)

    while True:
        raw = input("\n냉장고 재료를 쉼표로 입력하세요: ").strip()
        if raw.lower() == "q":
            print("종료합니다.")
            break

        ingredients = [i.strip() for i in raw.split(",") if i.strip()]
        if not ingredients:
            print("재료를 하나 이상 입력해주세요.")
            continue

        uid_raw = input("사용자 ID (없으면 Enter): ").strip()
        user_id = int(uid_raw) if uid_raw.isdigit() else None

        score_raw = input("최소 당뇨 적합도 점수 (기본 30, 0-100): ").strip()
        min_score = float(score_raw) if score_raw else 30.0

        results = model.recommend(
            ingredients=ingredients,
            user_id=user_id,
            top_n=10,
            min_diabetes_score=min_score,
        )

        if results.empty:
            print("조건에 맞는 레시피가 없습니다. 최소 점수를 낮춰보세요.")
            continue

        print(f"\n{'순위':<4} {'레시피 이름':<40} {'적합도':>6} {'등급':<8} {'칼로리':>7} {'최종점수':>8}")
        print("-" * 80)
        for rank, row in results.iterrows():
            name = row["name"][:38]
            print(
                f"{rank+1:<4} {name:<40} "
                f"{row['diabetes_score']:>6.1f} {row['diabetes_label']:<8} "
                f"{row['calories']:>7.0f} {row['final_score']:>8.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="당뇨 레시피 추천 ML 시스템")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE, help="archive.zip 경로")
    parser.add_argument("--max-recipes", type=int, default=50_000)
    parser.add_argument("--no-cf", action="store_true", help="협업 필터링 비활성화")
    parser.add_argument(
        "--ingredients", nargs="+",
        help="재료 목록 (비대화형 모드). 예: --ingredients chicken garlic broccoli"
    )
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=30.0)
    args = parser.parse_args()

    model = build_recommender(
        archive_path=args.archive,
        max_recipes=args.max_recipes,
        use_cf=not args.no_cf,
    )

    if args.ingredients:
        results = model.recommend(
            ingredients=args.ingredients,
            user_id=args.user_id,
            top_n=args.top_n,
            min_diabetes_score=args.min_score,
        )
        print(results.to_string(index=False))
    else:
        interactive_session(model)


if __name__ == "__main__":
    main()
