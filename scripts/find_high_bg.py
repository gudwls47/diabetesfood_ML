"""
BG rise 50 mg/dL 이상을 유발하는 레시피/재료 조합 탐색
"""
import sys
sys.path.insert(0, r'C:\Users\gudwl\diabetes_ml')
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from src.data_loader import load_foodcom_recipes
from src.bg_model import BGRiseModel

ARCHIVE  = r'C:\Users\gudwl\Downloads\archive.zip'
CGMACROS = r'C:\Users\gudwl\Downloads\CGMacros_dateshifted365.zip'
PRE_BG   = 130.0

print("레시피 로딩 중 (10만개)...")
df = load_foodcom_recipes(ARCHIVE, nrows=100000)
print(f"  -> {len(df):,}개")

print("모델 학습 중...")
model = BGRiseModel()
model.fit(cgmacros_path=CGMACROS)
print(f"  -> 완료 ({model._n_samples}건 학습)")

# 식사 유형별로 BG rise 예측
results = []
for meal_type in ("breakfast", "lunch", "dinner"):
    for _, row in df.iterrows():
        nutr = {
            "carbs_g":     float(row.get("carbs_g",     0) or 0),
            "protein_g":   float(row.get("protein_g",   0) or 0),
            "fat_g":       float(row.get("fat_g",       0) or 0),
            "fiber_g":     float(row.get("fiber_g",     0) or 0),
            "net_carbs_g": float(row.get("net_carbs_g", 0) or 0),
            "calories":    float(row.get("calories",    0) or 0),
        }
        bg_rise = model.predict_from_nutrition(nutr, meal_type=meal_type, pre_bg=PRE_BG)
        if bg_rise >= 50:
            results.append({
                "name":      row["name"],
                "meal_type": meal_type,
                "bg_rise":   round(bg_rise, 1),
                "carbs_g":   round(nutr["carbs_g"], 1),
                "protein_g": round(nutr["protein_g"], 1),
                "fat_g":     round(nutr["fat_g"], 1),
                "calories":  round(nutr["calories"], 1),
            })
    print(f"  {meal_type}: {sum(1 for r in results if r['meal_type']==meal_type)}개 해당")

res_df = pd.DataFrame(results).sort_values("bg_rise", ascending=False).drop_duplicates("name")

print(f"\n=== 식전 혈당 {PRE_BG:.0f} mg/dL 기준 BG rise 50+ 레시피 ===")
print(f"총 {len(res_df)}개 (중복 제거)\n")

print(f"{'레시피명':35s}  {'식사':9s}  {'BG rise':>8s}  {'탄수화물':>8s}  {'단백질':>7s}  {'지방':>6s}  {'칼로리':>7s}")
print("-" * 90)
for _, r in res_df.head(30).iterrows():
    print(f"{str(r['name'])[:35]:35s}  {r['meal_type']:9s}  "
          f"{r['bg_rise']:>7.1f}  {r['carbs_g']:>7.1f}g  "
          f"{r['protein_g']:>6.1f}g  {r['fat_g']:>5.1f}g  {r['calories']:>6.0f}")

# 영양성분 분포 분석
print(f"\n=== BG rise 50+ 레시피 영양성분 분포 ===")
print(res_df[["bg_rise","carbs_g","protein_g","fat_g","calories"]].describe().round(1).to_string())
