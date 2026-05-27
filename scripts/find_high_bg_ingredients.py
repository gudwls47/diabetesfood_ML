"""
BG rise 80+ mg/dL 레시피의 재료 분석
"""
import sys
sys.path.insert(0, r'C:\Users\gudwl\diabetes_ml')
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from src.data_loader import load_foodcom_recipes
from src.bg_model import BGRiseModel

ARCHIVE  = r'C:\Users\gudwl\Downloads\archive.zip'
CGMACROS = r'C:\Users\gudwl\Downloads\CGMacros_dateshifted365.zip'

print("로딩 중...")
df = load_foodcom_recipes(ARCHIVE, nrows=100000)
model = BGRiseModel()
model.fit(cgmacros_path=CGMACROS)

results = []
for _, row in df.iterrows():
    nutr = {k: float(row.get(k, 0) or 0)
            for k in ("carbs_g","protein_g","fat_g","fiber_g","net_carbs_g","calories")}
    bg = model.predict_from_nutrition(nutr, meal_type="breakfast", pre_bg=130)
    if bg >= 80:
        results.append({
            "name":        row["name"],
            "bg_rise":     round(bg, 1),
            "carbs_g":     round(nutr["carbs_g"], 1),
            "calories":    round(nutr["calories"], 1),
            "ingredients": row["ingredients"],
            "tags":        row["tags"],
        })

res = pd.DataFrame(results).sort_values("bg_rise", ascending=False)
print(f"\nBG rise 80+ 레시피: {len(res)}개\n")

# 상위 15개 재료 출력
print("=== 상위 15개 레시피 재료 ===")
for _, r in res.head(15).iterrows():
    ingr = r['ingredients'][:5] if isinstance(r['ingredients'], list) else []
    print(f"\n[{r['bg_rise']:.0f} mg/dL] {str(r['name'])[:45]}  (탄수화물 {r['carbs_g']:.0f}g)")
    print(f"  재료: {', '.join(ingr)}")

# 재료 빈도 집계
from collections import Counter
all_ingr = []
for ingr_list in res["ingredients"]:
    if isinstance(ingr_list, list):
        all_ingr.extend(ingr_list)
cnt = Counter(all_ingr).most_common(30)
print("\n=== 고위험 레시피 공통 재료 Top 30 ===")
for ingr, c in cnt:
    print(f"  {ingr:35s}: {c}회")
