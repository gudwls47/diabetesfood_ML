"""
알코올·칵테일 필터로 제거된 레시피 목록 확인
"""
import sys, ast, zipfile
sys.path.insert(0, r'C:\Users\gudwl\diabetes_ml')
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from src.data_loader import _EXCLUDE_TAGS, _EXCLUDE_NAME_KEYWORDS, _EXCLUDE_NAME_RE

ARCHIVE = r'C:\Users\gudwl\Downloads\archive.zip'

with zipfile.ZipFile(ARCHIVE) as z:
    with z.open('RAW_recipes.csv') as f:
        df = pd.read_csv(f, nrows=100000)

def parse_list(s):
    try: return ast.literal_eval(s)
    except: return []

df['tags'] = df['tags'].apply(parse_list)

# 태그 기반 제거 대상
tag_hit = df['tags'].apply(lambda tags: bool(set(tags) & _EXCLUDE_TAGS))

# 이름 키워드 기반 제거 대상 (단어 경계 정규식)
name_hit_mask = df['name'].apply(lambda n: bool(_EXCLUDE_NAME_RE.search(str(n))))

filtered = df[tag_hit | name_hit_mask].copy()
filtered['hit_reason'] = ''
filtered.loc[tag_hit[tag_hit].index, 'hit_reason'] = '태그'
filtered.loc[name_hit_mask[name_hit_mask].index, 'hit_reason'] = '이름 키워드'

print(f"총 제거 레시피: {len(filtered):,}개 (10만개 기준)\n")

# 태그별 집계
from collections import Counter
all_tags = []
for tags in filtered['tags']:
    all_tags.extend([t for t in tags if t in _EXCLUDE_TAGS])
print("=== 제거 태그 분포 ===")
for tag, cnt in Counter(all_tags).most_common():
    print(f"  {tag:35s}: {cnt:,}개")

# 이름 키워드별 집계
print("\n=== 제거 키워드 분포 ===")
kw_cnt = Counter()
for name in filtered['name']:
    m = _EXCLUDE_NAME_RE.search(str(name))
    if m:
        kw_cnt[m.group(0).lower()] += 1
for kw, cnt in kw_cnt.most_common():
    print(f"  {kw:25s}: {cnt:,}개")

# 샘플 레시피 출력
print("\n=== 제거된 레시피 샘플 (50개) ===")
for _, row in filtered.sample(n=min(50, len(filtered)), random_state=42).sort_values('name').iterrows():
    m = _EXCLUDE_NAME_RE.search(str(row['name']))
    hit_kw = m.group(0) if m else None
    hit_tag = next((t for t in row['tags'] if t in _EXCLUDE_TAGS), None)
    reason = f"키워드: '{hit_kw}'" if hit_kw else f"태그: '{hit_tag}'"
    print(f"  {str(row['name'])[:50]:50s}  ({reason})")
