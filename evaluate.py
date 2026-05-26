"""
모델 평가 스크립트.

평가 항목
---------
1. 혈당 상승 예측 모델 (RandomForestRegressor): 5-fold CV RMSE
2. 레시피 추천 적합성 분포: 당뇨 적합 / 부적합 비율 (샘플 레시피 기준)
"""

import argparse
import pandas as pd

from src.data_loader import load_indian_recipes
from src.nutrition_db import load_nutrition_db, estimate_recipe_nutrition
from src.bg_model import BGRiseModel

DEFAULT_FOODDATA1 = r"C:\Users\gudwl\Downloads\archive (1).zip"
DEFAULT_ARCHIVE2  = r"C:\Users\gudwl\Downloads\archive (2).zip"
DEFAULT_ARCHIVE3  = r"C:\Users\gudwl\Downloads\archive (3).zip"


def evaluate_bg_model(model: BGRiseModel):
    """학습된 BG 모델의 CV RMSE를 출력."""
    print("\n[1] 혈당 상승 예측 모델 (RandomForestRegressor)")
    print("-" * 50)
    print(f"  학습 샘플 수    : {model._n_samples}건")
    print(f"  CV RMSE (5-fold): {model.cv_rmse:.2f} mg/dL")
    print(f"  해석: 실제 혈당 상승값 대비 평균 +-{model.cv_rmse:.1f} mg/dL 오차")


def evaluate_recommendations(
    recipes_df: pd.DataFrame,
    nutrition_db: pd.DataFrame,
    model: BGRiseModel,
    n_queries: int = 200,
    pre_bg: float = 110.0,
    meal_type: str = "lunch",
):
    """샘플 레시피에 대한 당뇨 적합성 분포를 출력."""
    print("\n[2] 레시피 추천 적합성 분포 평가")
    print("-" * 50)

    sample = recipes_df.sample(n=min(n_queries, len(recipes_df)), random_state=42)

    labels = []
    for _, row in sample.iterrows():
        nutr = estimate_recipe_nutrition(row["ingredients"], nutrition_db)
        bg_rise = model.predict(nutr, meal_type, pre_bg)
        label, _, _ = model.classify(bg_rise, pre_bg)
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

    print("데이터 로딩 및 모델 학습 중 ...")
    recipes      = load_indian_recipes(args.archive3, archive2_path=args.archive2)
    nutrition_db = load_nutrition_db(args.fooddata1)
    print(f"  레시피: {len(recipes):,}개  |  영양 DB: {len(nutrition_db):,}개 식품")

    bg_model = BGRiseModel()
    bg_model.fit(args.archive2, nutrition_db)
    print("  BG 모델 학습 완료\n")

    evaluate_bg_model(bg_model)
    evaluate_recommendations(
        recipes, nutrition_db, bg_model,
        n_queries=args.n_queries,
        pre_bg=args.pre_bg,
    )
    print()
