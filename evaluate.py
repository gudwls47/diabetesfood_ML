"""
모델 평가 스크립트.

평가 항목
---------
1. CGMacros 데이터 통계 (참여자 수, 식사 수, BG rise 분포)
2. 5-fold 교차검증 RMSE
3. 특성 중요도
4. 레시피 추천 적합성 분포 (당뇨 적합 / 부적합 비율)
"""

import argparse
import pandas as pd

from src.data_loader import load_indian_recipes
from src.nutrition_db import load_nutrition_db, estimate_recipe_nutrition
from src.bg_model import BGRiseModel
from src.cgmacros_loader import load_cgmacros

DEFAULT_FOODDATA1 = r"C:\Users\gudwl\Downloads\archive (1).zip"
DEFAULT_ARCHIVE3  = r"C:\Users\gudwl\Downloads\archive (3).zip"
DEFAULT_CGMACROS  = r"C:\Users\gudwl\Downloads\CGMacros_dateshifted365.zip"


def show_data_stats(cgmacros_path: str):
    """CGMacros 데이터 통계를 출력합니다."""
    print("\n[1] CGMacros 데이터 통계")
    print("-" * 50)
    df = load_cgmacros(cgmacros_path)
    s = df["bg_rise"]
    print(f"  총 식사 기록    : {len(df)}건")
    print(f"  BG rise 범위   : {s.min():.1f} ~ {s.max():.1f} mg/dL")
    print(f"  평균 / 중간값   : {s.mean():.1f} / {s.median():.1f} mg/dL")
    print()
    print(f"  식사 유형별 분포:")
    for mtype, cnt in df["meal_type"].value_counts().items():
        print(f"    {mtype:12s}: {cnt}건  "
              f"(BG rise 평균 {df.loc[df['meal_type']==mtype,'bg_rise'].mean():.1f} mg/dL)")
    print()
    print(f"  탄수화물 분포 (순탄수화물):")
    q = df["net_carbs_g"].describe()
    print(f"    평균 {q['mean']:.1f}g  |  중간값 {q['50%']:.1f}g  |  "
          f"범위 {q['min']:.1f}~{q['max']:.1f}g")
    return df


def show_cv_rmse(model: BGRiseModel, cgmacros_path: str):
    """5-fold 교차검증 RMSE를 출력합니다."""
    print("\n[2] 5-fold 교차검증 성능")
    print("-" * 50)
    print("  계산 중...", end="", flush=True)
    mean_rmse, std_rmse = model.cv_rmse(cgmacros_path)
    print(f"\r  RMSE: {mean_rmse:.2f} +/- {std_rmse:.2f} mg/dL")
    print(f"  (낮을수록 좋음 — 혈당 예측 오차)")


def show_feature_importance(model: BGRiseModel):
    """특성 중요도를 출력합니다."""
    print("\n[3] 특성 중요도 (BG rise 예측에 미치는 영향)")
    print("-" * 50)
    fi = model.feature_importances()
    for feat, imp in fi.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:15s}: {imp:.3f}  {bar}")


def evaluate_recommendations(
    recipes_df: pd.DataFrame,
    nutrition_db: pd.DataFrame,
    model: BGRiseModel,
    n_queries: int = 200,
    pre_bg: float = 110.0,
    meal_type: str = "lunch",
):
    """샘플 레시피에 대한 당뇨 적합성 분포를 출력합니다."""
    print(f"\n[4] 레시피 추천 적합성 분포 (식전 혈당 {pre_bg:.0f} mg/dL 기준)")
    print("-" * 50)

    sample = recipes_df.sample(n=min(n_queries, len(recipes_df)), random_state=42)

    labels = []
    for _, row in sample.iterrows():
        nutr = estimate_recipe_nutrition(row["ingredients"], nutrition_db)
        bg_rise = model.predict_from_nutrition(nutr, meal_type=meal_type, pre_bg=pre_bg)
        label, _, _ = model.classify(bg_rise, pre_bg)
        labels.append(label)

    total      = len(labels)
    suitable   = labels.count("적합")
    unsuitable = labels.count("부적합")

    print(f"  평가 레시피 수  : {total}개")
    print(f"  적합            : {suitable}개 ({suitable / total * 100:.1f}%)")
    print(f"  부적합          : {unsuitable}개 ({unsuitable / total * 100:.1f}%)")
    print(f"  판정 기준       : 식후 2시간 혈당 180 mg/dL 미만 (대한당뇨병학회)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="당뇨 레시피 추천 모델 평가")
    parser.add_argument("--fooddata1",  default=DEFAULT_FOODDATA1)
    parser.add_argument("--archive3",   default=DEFAULT_ARCHIVE3)
    parser.add_argument("--cgmacros",   default=DEFAULT_CGMACROS)
    parser.add_argument("--pre-bg",     type=float, default=110.0)
    parser.add_argument("--n-queries",  type=int,   default=200)
    parser.add_argument("--skip-cv",    action="store_true",
                        help="교차검증 생략 (느림)")
    args = parser.parse_args()

    print("데이터 로딩 중 ...")
    recipes      = load_indian_recipes(args.archive3)
    nutrition_db = load_nutrition_db(args.fooddata1)
    print(f"  레시피: {len(recipes):,}개  |  영양 DB: {len(nutrition_db):,}개 식품")

    bg_model = BGRiseModel()

    show_data_stats(args.cgmacros)

    print("\n[모델 학습 중 ...]")
    bg_model.fit(cgmacros_path=args.cgmacros)
    print(f"  완료: {bg_model._n_samples}건 학습")

    if not args.skip_cv:
        show_cv_rmse(bg_model, args.cgmacros)

    show_feature_importance(bg_model)
    evaluate_recommendations(
        recipes, nutrition_db, bg_model,
        n_queries=args.n_queries,
        pre_bg=args.pre_bg,
    )
    print()
