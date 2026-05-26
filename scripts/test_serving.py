import sys
sys.path.insert(0, r'C:\Users\gudwl\diabetes_ml')
sys.stdout.reconfigure(encoding='utf-8')

from src.data_loader import load_foodcom_recipes

df = load_foodcom_recipes(r'C:\Users\gudwl\Downloads\archive.zip', nrows=50000)

print(f"레시피 수: {len(df):,}")
print(f"칼로리  - 평균: {df['calories'].mean():.0f}  중간값: {df['calories'].median():.0f}  최대: {df['calories'].max():.0f}")
print(f"탄수화물 - 평균: {df['carbs_g'].mean():.1f}g  중간값: {df['carbs_g'].median():.1f}g  최대: {df['carbs_g'].max():.1f}g")
print(f"단백질   - 평균: {df['protein_g'].mean():.1f}g  중간값: {df['protein_g'].median():.1f}g")
print(f"지방     - 평균: {df['fat_g'].mean():.1f}g  중간값: {df['fat_g'].median():.1f}g")

# 이름으로 확인
targets = ['fluffy white rice', 'seasoned rice', 'cup a rice',
           'poached chicken', 'two ingredient zucchini', 'broth simmered']
print("\n=== 샘플 레시피 ===")
for t in targets:
    rows = df[df['name'].str.contains(t, case=False, na=False)]
    if not rows.empty:
        r = rows.iloc[0]
        print(f"  {str(r['name'])[:38]:38s}  cal={r['calories']:5.0f}  carb={r['carbs_g']:5.1f}g  prot={r['protein_g']:4.1f}g")
