import zipfile, pandas as pd, ast, sys
sys.stdout.reconfigure(encoding='utf-8')

with zipfile.ZipFile(r'C:\Users\gudwl\Downloads\archive.zip') as z:
    with z.open('RAW_recipes.csv') as f:
        df = pd.read_csv(f, nrows=5000)

def parse(s):
    try:
        v = ast.literal_eval(s)
        return v if len(v) >= 7 else None
    except:
        return None

df['n']    = df['nutrition'].apply(parse)
df         = df[df['n'].notna()]
df['cal']  = df['n'].apply(lambda x: x[0])
df['carb'] = df['n'].apply(lambda x: x[6] * 2.75)

targets = ['fluffy white rice', 'seasoned rice', 'cup a rice',
           'poached chicken breast', 'two ingredient zucchini',
           'simple crock pot chicken', 'broth simmered rice']

for t in targets:
    rows = df[df['name'].str.contains(t, case=False, na=False)]
    if not rows.empty:
        r = rows.iloc[0]
        name = str(r['name'])[:40]
        cal  = r['cal']
        carb = r['carb']
        norm_ratio = min(1.0, 600.0 / max(cal, 1.0))
        print(f"{name:40s}  cal={cal:6.0f}  carb={carb:5.1f}g  -> 600kcal 기준: {carb*norm_ratio:.1f}g")
