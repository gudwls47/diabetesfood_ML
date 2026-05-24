"""
당뇨 환자를 위한 레시피 추천 시스템 — 실행 진입점

흐름
----
1. 냉장고 재료 입력
2. 그 재료로 만들 수 있는 레시피 필터링
3. 레시피별 혈당 상승 추정값(mg/dL) 계산
4. 당뇨 적합 여부 판단 및 출력
"""

import argparse
from src.data_loader import load_recipes, load_interactions
from src.recommender import DiabetesRecipeRecommender

DEFAULT_ARCHIVE = r"C:\Users\gudwl\Downloads\archive.zip"


def build_recommender(archive_path: str, max_recipes: int = 50_000, use_cf: bool = True):
    print(f"[1/3] 레시피 로딩 중 (최대 {max_recipes:,}개) ...")
    recipes = load_recipes(archive_path, nrows=max_recipes)
    print(f"      -> {len(recipes):,}개 로드 완료")

    interactions = None
    if use_cf:
        print("[2/3] 사용자 평점 데이터 로딩 중 ...")
        interactions = load_interactions(archive_path)
        print(f"      -> {len(interactions):,}건 로드 완료")
    else:
        print("[2/3] 협업 필터링 생략")

    print("[3/3] 모델 학습 중 ...")
    model = DiabetesRecipeRecommender(n_factors=50)
    model.fit(recipes, interactions)
    print("      -> 완료!\n")
    return model


def _print_results(results):
    if results.empty:
        print("\n조건에 맞는 레시피가 없습니다.")
        print("  → 재료를 더 추가하거나 min_coverage 값을 낮춰 보세요.")
        return

    print(f"\n{'':=<72}")
    print(f"  추천 레시피 {len(results)}개")
    print(f"{'':=<72}")

    for i, row in results.iterrows():
        print(f"\n[{i+1}] {row['name']}")
        print(f"     재료 보유율  : {row['coverage']*100:.0f}%")
        print(f"     탄수화물     : {row['carbs_g']:.1f}g  |  당류: {row['sugar_g']:.1f}g  |  칼로리: {row['calories']:.0f} kcal")
        print(f"     혈당 상승 예측: +{row['bg_rise_mg_dl']:.0f} mg/dL  (추정치, 개인차 있음)")
        print(f"     당뇨 적합성  : [{row['suitability']}]  {row['suitability_desc']}")

    print(f"\n{'':=<72}")
    print("※ 혈당 예측은 탄수화물 기반 추정값입니다. 실제 수치는 개인마다 다릅니다.")
    print("  (ADA Standards of Care 2025 — 1회 식사 탄수화물 45-60g 기준 적용)")


def interactive_session(model: DiabetesRecipeRecommender):
    print("=" * 72)
    print("  당뇨 환자를 위한 냉장고 레시피 추천기")
    print("  종료하려면 'q' 입력")
    print("=" * 72)

    while True:
        raw = input("\n냉장고 재료를 쉼표로 입력: ").strip()
        if raw.lower() == "q":
            print("종료합니다.")
            break

        ingredients = [i.strip() for i in raw.split(",") if i.strip()]
        if not ingredients:
            print("재료를 하나 이상 입력해주세요.")
            continue

        uid_raw = input("사용자 ID (개인화 추천, 없으면 Enter): ").strip()
        user_id = int(uid_raw) if uid_raw.isdigit() else None

        cov_raw = input("최소 재료 보유율 % (기본 50): ").strip()
        min_cov = float(cov_raw) / 100 if cov_raw else 0.5

        excl_raw = input("제외할 등급 (예: 비권장, 없으면 Enter): ").strip()
        exclude = [e.strip() for e in excl_raw.split(",") if e.strip()] or None

        results = model.recommend(
            ingredients=ingredients,
            user_id=user_id,
            top_n=10,
            min_coverage=min_cov,
            exclude_suitability=exclude,
        )
        _print_results(results)


def main():
    parser = argparse.ArgumentParser(description="당뇨 레시피 추천 ML 시스템")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--max-recipes", type=int, default=50_000)
    parser.add_argument("--no-cf", action="store_true")
    parser.add_argument("--ingredients", nargs="+")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args()

    model = build_recommender(args.archive, args.max_recipes, not args.no_cf)

    if args.ingredients:
        results = model.recommend(
            ingredients=args.ingredients,
            user_id=args.user_id,
            top_n=args.top_n,
            min_coverage=args.min_coverage,
        )
        _print_results(results)
    else:
        interactive_session(model)


if __name__ == "__main__":
    main()
