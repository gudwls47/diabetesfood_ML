"""
모델 평가 스크립트.

평가 항목
---------
1. 혈당 예측 공식 파라미터 출력
2. 레시피 추천 적합성 분포: 당뇨 적합 / 부적합 비율 (샘플 레시피 기준)
"""

import argparse
import pandas as pd

from src.data_loader import load_indian_recipes
from src.nutrition_db import load_nutrition_db, estimate_recipe_nutrition
from src.bg_model import BGRiseModel, BASE_RISE, CARB_FACTOR, MAX_RISE

DEFAULT_FOODDATA1 = r"C:\Users\gudwl\Downloads\archive (1).zip"
DEFAULT_ARCHIVE2  = r"C:\Users\gudwl\Downloads\archive (2).zip"
DEFAULT_ARCHIVE3  = r"C:\Users\gudwl\Downloads\archive (3).zip"


def show_formula():
    """혈당 예측 공식 파라미터를 출력합니다."""
    print("\n[1] 혈당 상승 예측 공식")
    print("-" * 50)
    print(f"  BG_rise = {BASE_RISE} + net_carbs_g x {CARB_FACTOR}  (상한 {MAX_RISE} mg/dL)")
    print(f"  예시)")
    examples = [(10, 110), (50, 110), (100, 110), (150, 120), (180, 120)]
    for nc, pre in examples:
        rise = min(MAX_RISE, BASE_RISE + nc * CARB_FACTOR)
        post = pre + rise
        label = "부적합" if post >= 180 else "적합"
        print(f"    net_carbs {nc:3g}g + 식전 {pre} -> +{rise:.0f} -> 식후 {post:.0f} [{label}]")


def evaluate_recommendations(
    recipes_df: pd.DataFrame,
    nutrition_db: pd.DataFrame,
    model: BGRiseModel,
    n_queries: int = 200,
    pre_bg: float = 110.0,
    meal_type: str = "lunch",
):
    """샘플 레시피에 대한 당뇨 적합성 분포를 출력합니다."""
    print("\n[2] 레시피 추천 적합성 분포 평가")
    print("-" * 50)

    sample = recipes_df.sample(n=min(n_queries, len(recipes_df)), random_state=42)

    labels = []
    for _, row in sample.iterrows():
        nutr   = estimate_recipe_nutrition(row["ingredients"], nutrition_db)
        rise   = model.predict(nutr, meal_type, pre_bg)
        label, _, _ = model.classify(rise, pre_bg)
        labels.append(label)

    total      = len(labels)
    suitable   = labels.count("적합")
    unsuitable = labels.count("부적합")

    print(f"  평가 레시피 수  : {total}개 (식전 혈당 {pre_bg:.0f} mg/dL 기준)")
    print(f"  적합            : {suitable}개 ({suitable / total * 100:.1f}%)")
    print(f"  부적합          : {unsuitable}개 ({unsuitable / total * 100:.1f}%)")
    print(f"  판정 기준       : 식후 2시간 혈당 180 mg/dL 미만 (대한당뇨병학회)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="당뇨 레시피 추천 모델 평가")
    parser.add_argument("--fooddata1", default=DEFAULT_FOODDATA1)
    parser.add_argument("--archive2",  default=DEFAULT_ARCHIVE2)
    parser.add_argument("--archive3",  default=DEFAULT_ARCHIVE3)
    parser.add_argument("--pre-bg",    type=float, default=110.0,
                        help="평가 기준 식전 혈당 (mg/dL)")
    parser.add_argument("--n-queries", type=int,   default=200,
                        help="적합성 평가에 사용할 레시피 수")
    args = parser.parse_args()

    print("데이터 로딩 중 ...")
    recipes      = load_indian_recipes(args.archive3, archive2_path=args.archive2)
    nutrition_db = load_nutrition_db(args.fooddata1)
    print(f"  레시피: {len(recipes):,}개  |  영양 DB: {len(nutrition_db):,}개 식품\n")

    bg_model = BGRiseModel()
    bg_model.fit()

    show_formula()
    evaluate_recommendations(
        recipes, nutrition_db, bg_model,
        n_queries=args.n_queries,
        pre_bg=args.pre_bg,
    )
    print()
