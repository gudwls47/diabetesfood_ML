"""
당뇨 환자를 위한 냉장고 레시피 추천기

사용법
------
  python main.py

재료를 한국어 또는 영어로 입력하면:
  1. 그 재료로 만들 수 있는 레시피 추천
  2. 레시피별 예상 혈당 상승값 (mg/dL) 계산
  3. 당뇨 적합 여부 판정
"""

import argparse
from src.data_loader import load_recipes, load_interactions
from src.nutrition_db import load_nutrition_db
from src.recommender import DiabetesRecipeRecommender

DEFAULT_ARCHIVE      = r"C:\Users\gudwl\Downloads\archive.zip"
DEFAULT_FOODDATA1    = r"C:\Users\gudwl\Downloads\archive (1).zip"
DEFAULT_ARCHIVE2     = r"C:\Users\gudwl\Downloads\archive (2).zip"


def build_recommender(
    archive_path: str,
    fooddata1_path: str,
    archive2_path: str,
    max_recipes: int = 50_000,
    use_cf: bool = True,
) -> DiabetesRecipeRecommender:

    print(f"[1/4] 레시피 로딩 중 (최대 {max_recipes:,}개) ...")
    recipes = load_recipes(archive_path, nrows=max_recipes)
    print(f"      -> {len(recipes):,}개 완료")

    print("[2/4] 영양성분 DB 로딩 중 ...")
    nutrition_db = load_nutrition_db(fooddata1_path)
    print(f"      -> {len(nutrition_db):,}개 식품 완료")

    interactions = None
    if use_cf:
        print("[3/4] 사용자 평점 데이터 로딩 중 ...")
        interactions = load_interactions(archive_path)
        print(f"      -> {len(interactions):,}건 완료")
    else:
        print("[3/4] 협업 필터링 생략")

    print("[4/4] 모델 학습 중 ...")
    model = DiabetesRecipeRecommender(n_factors=50)
    model.fit(
        recipes_df=recipes,
        nutrition_db=nutrition_db,
        archive2_path=archive2_path,
        interactions_df=interactions,
    )
    print("      -> 완료!\n")
    return model


def _print_results(results):
    if results.empty:
        print("\n조건에 맞는 레시피가 없습니다.")
        print("  -> 재료를 더 추가하거나 최소 보유율을 낮춰 보세요.")
        return

    print(f"\n{'='*65}")
    print(f"  추천 레시피 {len(results)}개")
    print(f"{'='*65}")

    for i, row in results.iterrows():
        print(f"\n[{i+1}] {row['name']}")
        print(f"     재료 보유율 : {row['coverage']*100:.0f}%")
        print(f"     탄수화물    : {row['carbs_g']:.1f}g  |  "
              f"식이섬유: {row['fiber_g']:.1f}g  |  "
              f"순 탄수화물: {row['net_carbs_g']:.1f}g")
        if row['bg_rise_mg_dl'] is not None:
            print(f"     혈당 상승   : +{row['bg_rise_mg_dl']:.0f} mg/dL (예측)")
        print(f"     당뇨 적합성 : [{row['suitability']}]  {row['suitability_desc']}")

    print(f"\n{'='*65}")
    print("※ 혈당 예측은 1인 CGM 데이터 기반 Random Forest 모델 추정치입니다.")
    print("  개인차가 크므로 참고용으로만 활용하세요.")


def interactive_session(model: DiabetesRecipeRecommender):
    print("=" * 65)
    print("  당뇨 환자를 위한 냉장고 레시피 추천기")
    print("  한국어/영어 재료 모두 입력 가능  |  종료: q")
    print("=" * 65)

    while True:
        raw = input("\n냉장고 재료 입력 (쉼표 구분): ").strip()
        if raw.lower() == "q":
            print("종료합니다.")
            break

        ingredients = [i.strip() for i in raw.split(",") if i.strip()]
        if not ingredients:
            print("재료를 하나 이상 입력해주세요.")
            continue

        meal = input("식사 유형 (breakfast/lunch/dinner/snacks, 기본 lunch): ").strip()
        meal = meal if meal else "lunch"

        cov = input("최소 재료 보유율 % (기본 50): ").strip()
        min_cov = float(cov) / 100 if cov else 0.5

        excl = input("제외할 등급 (예: 비권장 / 없으면 Enter): ").strip()
        exclude = [e.strip() for e in excl.split(",") if e.strip()] or None

        results = model.recommend(
            ingredients=ingredients,
            top_n=10,
            min_coverage=min_cov,
            meal_type=meal,
            exclude_suitability=exclude,
        )
        _print_results(results)


def main():
    parser = argparse.ArgumentParser(description="당뇨 레시피 추천 ML 시스템")
    parser.add_argument("--archive",    default=DEFAULT_ARCHIVE)
    parser.add_argument("--fooddata1",  default=DEFAULT_FOODDATA1)
    parser.add_argument("--archive2",   default=DEFAULT_ARCHIVE2)
    parser.add_argument("--max-recipes", type=int, default=50_000)
    parser.add_argument("--no-cf",       action="store_true")
    parser.add_argument("--ingredients", nargs="+")
    parser.add_argument("--meal-type",   default="lunch")
    parser.add_argument("--top-n",       type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args()

    model = build_recommender(
        args.archive, args.fooddata1, args.archive2,
        args.max_recipes, not args.no_cf,
    )

    if args.ingredients:
        results = model.recommend(
            ingredients=args.ingredients,
            top_n=args.top_n,
            min_coverage=args.min_coverage,
            meal_type=args.meal_type,
        )
        _print_results(results)
    else:
        interactive_session(model)


if __name__ == "__main__":
    main()
