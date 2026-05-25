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
from src.data_loader import load_recipes, load_indian_recipes, load_interactions
from src.nutrition_db import load_nutrition_db
from src.recommender import DiabetesRecipeRecommender

DEFAULT_ARCHIVE      = r"C:\Users\gudwl\Downloads\archive.zip"
DEFAULT_FOODDATA1    = r"C:\Users\gudwl\Downloads\archive (1).zip"
DEFAULT_ARCHIVE2     = r"C:\Users\gudwl\Downloads\archive (2).zip"
DEFAULT_ARCHIVE3     = r"C:\Users\gudwl\Downloads\archive (3).zip"  # 인도 레시피


def build_recommender(
    archive_path: str,
    fooddata1_path: str,
    archive2_path: str,
    archive3_path: str = None,
    max_recipes: int = 50_000,
    use_cf: bool = True,
) -> DiabetesRecipeRecommender:

    print(f"[1/4] 레시피 로딩 중 ...")
    if archive3_path:
        recipes = load_indian_recipes(archive3_path, nrows=max_recipes)
        print(f"      -> 인도 레시피 {len(recipes):,}개 완료 (혈당 예측 정확도 향상)")
    else:
        recipes = load_recipes(archive_path, nrows=max_recipes)
        print(f"      -> Food.com 레시피 {len(recipes):,}개 완료")

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
        print(f"     재료 보유율    : {row['coverage']*100:.0f}%")
        print(f"     탄수화물       : {row['carbs_g']:.1f}g  |  "
              f"식이섬유: {row['fiber_g']:.1f}g  |  "
              f"순 탄수화물: {row['net_carbs_g']:.1f}g")
        if row['bg_rise_mg_dl'] is not None:
            print(f"     혈당 상승 예측 : +{row['bg_rise_mg_dl']:.0f} mg/dL  "
                  f"-> 식후 혈당 {row['post_meal_bg']:.0f} mg/dL")
        print(f"     당뇨 적합성    : [{row['suitability']}]  {row['suitability_desc']}")

    print(f"\n{'='*65}")
    print("※ 판정 기준: 식후 2시간 혈당 180 mg/dL 미만 (대한당뇨병학회)")
    print("  1인 CGM 데이터 기반 추정치 — 개인차가 크므로 참고용으로만 활용하세요.")


def _input_pre_bg() -> float:
    """식전 혈당 입력 및 유효성 검사. 정상 범위(80~130) 벗어나면 경고."""
    while True:
        raw = input("식전 혈당 입력 (mg/dL, 80~130): ").strip()
        if not raw:
            print("  식전 혈당을 입력해주세요.")
            continue
        try:
            val = float(raw)
        except ValueError:
            print("  숫자로 입력해주세요. (예: 110)")
            continue
        if val < 50 or val > 400:
            print("  입력값이 너무 크거나 작습니다. 다시 입력해주세요.")
            continue
        if val < 80 or val > 130:
            print(f"  참고: 입력하신 {val:.0f} mg/dL은 식전 혈당 목표 범위(80~130)를 벗어납니다.")
        return val


def interactive_session(model: DiabetesRecipeRecommender):
    print("=" * 65)
    print("  당뇨 환자를 위한 냉장고 레시피 추천기")
    print("  한국어/영어 재료 모두 입력 가능  |  종료: q")
    print("=" * 65)

    while True:
        # 1. 식전 혈당
        print()
        pre_bg_raw = input("식전 혈당 입력 (mg/dL, 80~130) 또는 'q' 종료: ").strip()
        if pre_bg_raw.lower() == "q":
            print("종료합니다.")
            break
        try:
            pre_bg = float(pre_bg_raw)
            if pre_bg < 50 or pre_bg > 400:
                print("  입력값이 범위를 벗어났습니다. 다시 시도해주세요.")
                continue
            if pre_bg < 80 or pre_bg > 130:
                print(f"  참고: {pre_bg:.0f} mg/dL은 식전 목표 범위(80~130)를 벗어납니다.")
        except ValueError:
            print("  숫자로 입력해주세요.")
            continue

        # 2. 냉장고 재료
        raw = input("냉장고 재료 입력 (쉼표 구분): ").strip()
        ingredients = [i.strip() for i in raw.split(",") if i.strip()]
        if not ingredients:
            print("재료를 하나 이상 입력해주세요.")
            continue

        # 3. 식사 유형
        meal = input("식사 유형 (breakfast/lunch/dinner/snacks, 기본 lunch): ").strip()
        meal = meal if meal else "lunch"

        cov = input("최소 재료 보유율 % (기본 50): ").strip()
        min_cov = float(cov) / 100 if cov else 0.5

        results = model.recommend(
            ingredients=ingredients,
            top_n=10,
            min_coverage=min_cov,
            meal_type=meal,
            pre_bg=pre_bg,
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
    parser.add_argument("--pre-bg",      type=float, default=None,
                        help="식전 혈당 (mg/dL). 미입력 시 대화형 모드에서 질문.")
    parser.add_argument("--meal-type",   default="lunch")
    parser.add_argument("--top-n",       type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args()

    model = build_recommender(
        args.archive, args.fooddata1, args.archive2,
        DEFAULT_ARCHIVE3,
        args.max_recipes, not args.no_cf,
    )

    if args.ingredients:
        pre_bg = args.pre_bg if args.pre_bg is not None else _input_pre_bg()
        results = model.recommend(
            ingredients=args.ingredients,
            top_n=args.top_n,
            min_coverage=args.min_coverage,
            meal_type=args.meal_type,
            pre_bg=pre_bg,
        )
        _print_results(results)
    else:
        interactive_session(model)


if __name__ == "__main__":
    main()
