import zipfile, pandas as pd, sys, ast
sys.stdout.reconfigure(encoding='utf-8')

with zipfile.ZipFile(r'C:\Users\gudwl\Downloads\archive.zip') as z:
    with z.open('RAW_recipes.csv') as f:
        df = pd.read_csv(f)

print(f'총 레시피 수: {len(df):,}개')

def parse_nutr(s):
    try: return ast.literal_eval(s)
    except: return None

df['nutr'] = df['nutrition'].apply(parse_nutr)
df = df[df['nutr'].notna()]
df['calories']    = df['nutr'].apply(lambda x: x[0])
df['carbs_g']     = df['nutr'].apply(lambda x: x[6]) * 2.75
df['fat_g']       = df['nutr'].apply(lambda x: x[1]) * 0.78
df['protein_g']   = df['nutr'].apply(lambda x: x[4]) * 0.50

print('\n=== 영양성분 통계 ===')
print(df[['calories','carbs_g','fat_g','protein_g']].describe().round(1).to_string())

diabetic = df['tags'].str.contains('diabetic', na=False).sum()
lowcarb  = df['tags'].str.contains('low-carb', na=False).sum()
lowsugar = df['tags'].str.contains('low-sugar', na=False).sum()
print(f'\ndiabetic 태그: {diabetic:,}개')
print(f'low-carb 태그: {lowcarb:,}개')
print(f'low-sugar 태그: {lowsugar:,}개')

print('\n=== 샘플 레시피 5개 ===')
for _, r in df.head(5).iterrows():
    ingr = ast.literal_eval(r['ingredients'])[:3]
    name = r['name'][:45]
    print(f'  {name:45s} | carbs:{r["carbs_g"]:5.1f}g | 재료: {ingr}')
