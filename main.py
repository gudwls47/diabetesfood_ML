"""
당뇨 환자를 위한 냉장고 레시피 추천기

사용법
------
  python main.py

식전 혈당과 재료를 한국어 또는 영어로 입력하면:
  1. 그 재료로 만들 수 있는 Food.com 레시피 추천 (231,637개)
  2. 레시피별 예상 혈당 상승값 (mg/dL) 예측
     (CGMacros 45명 실측 데이터로 학습한 Gradient Boosting 모델)
  3. 대한당뇨병학회 기준으로 당뇨 적합 여부 판정
     (식후 2시간 혈당 180 mg/dL 미만 -> 적합)
"""

import argparse
from src.data_loader import load_foodcom_recipes
from src.recommender import DiabetesRecipeRecommender

DEFAULT_ARCHIVE_FOOD = r"C:\Users\gudwl\Downloads\archive.zip"
DEFAULT_CGMACROS     = r"C:\Users\gudwl\Downloads\CGMacros_dateshifted365.zip"


def build_recommender(
    archive_food_path: str,
    cgmacros_path: str,
    diabetic_only: bool = False,
) -> DiabetesRecipeRecommender:

    print("[1/2] Food.com 레시피 로딩 중 ...")
    recipes = load_foodcom_recipes(archive_food_path, diabetic_only=diabetic_only)
    print(f"      -> {len(recipes):,}개 완료")

    print("[2/2] CGMacros 데이터로 BG 예측 모델 학습 중 ...")
    model = DiabetesRecipeRecommender()
    model.fit(
        recipes_df=recipes,
        cgmacros_path=cgmacros_path,
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
        missing = row.get("missing_ingredients", [])
        if missing:
            missing_str = ", ".join(missing[:5])
            if len(missing) > 5:
                missing_str += f" 외 {len(missing)-5}개"
            print(f"     추가 필요 재료 : {missing_str}")
        print(f"     칼로리         : {row['calories']:.0f} kcal")
        print(f"     탄수화물       : {row['carbs_g']:.1f}g  |  "
              f"단백질: {row['protein_g']:.1f}g  |  "
              f"지방: {row['fat_g']:.1f}g")
        if row["bg_rise_mg_dl"] is not None:
            print(f"     혈당 상승 예측 : +{row['bg_rise_mg_dl']:.0f} mg/dL  "
                  f"-> 식후 혈당 {row['post_meal_bg']:.0f} mg/dL")
        print(f"     당뇨 적합성    : [{row['suitability']}]  {row['suitability_desc']}")

    print(f"\n{'='*65}")
    print("※ 판정 기준: 식후 2시간 혈당 180 mg/dL 미만 (대한당뇨병학회)")
    print("  CGMacros 45명 실측 데이터 기반 -- 개인차가 크므로 참고용으로만 활용하세요.")


def interactive_session(model: DiabetesRecipeRecommender):
    print("=" * 65)
    print("  당뇨 환자를 위한 냉장고 레시피 추천기")
    print("  한국어/영어 재료 모두 입력 가능  |  종료: q")
    print("=" * 65)

    while True:
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

        raw = input("냉장고 재료 입력 (쉼표 구분): ").strip()
        ingredients = [i.strip() for i in raw.split(",") if i.strip()]
        if not ingredients:
            print("재료를 하나 이상 입력해주세요.")
            continue

        meal = input("식사 유형 (breakfast/lunch/dinner, 기본 lunch): ").strip()
        meal = meal if meal in ("breakfast", "lunch", "dinner") else "lunch"

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
    parser.add_argument("--archive-food",  default=DEFAULT_ARCHIVE_FOOD,
                        help="Food.com archive.zip 경로")
    parser.add_argument("--cgmacros",      default=DEFAULT_CGMACROS)
    parser.add_argument("--diabetic-only", action="store_true",
                        help="당뇨/저탄수화물 태그 레시피만 로드 (~45,000개)")
    parser.add_argument("--ingredients",   nargs="+")
    parser.add_argument("--pre-bg",        type=float, default=None)
    parser.add_argument("--meal-type",     default="lunch")
    parser.add_argument("--top-n",         type=int,   default=10)
    parser.add_argument("--min-coverage",  type=float, default=0.5)
    args = parser.parse_args()

    model = build_recommender(
        args.archive_food,
        args.cgmacros,
        diabetic_only=args.diabetic_only,
    )

    if args.ingredients:
        pre_bg = args.pre_bg if args.pre_bg is not None else float(
            input("식전 혈당 입력 (mg/dL): ").strip()
        )
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
