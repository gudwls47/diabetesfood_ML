# 당뇨 환자를 위한 레시피 추천 시스템

냉장고 재료를 입력하면 당뇨 환자에게 적합한 레시피를 추천해주는 ML 시스템입니다.

## 모델 구성

| 단계 | 방식 | 설명 |
|------|------|------|
| 1 | **콘텐츠 기반 필터링** | TF-IDF + 코사인 유사도로 재료 매칭 |
| 2 | **당뇨 적합도 스코어링** | 당분/탄수화물 %DV 기반 0-100점 산정 |
| 3 | **협업 필터링 (SVD)** | 사용자 평점 데이터로 개인화 추천 |

## 데이터셋

- **[Food.com Recipes (Kaggle)](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)** (`archive.zip`)
  - `RAW_recipes.csv` : 약 23만 개 레시피 (재료, 영양정보, 조리법)
  - `interactions_train.csv` : 약 100만 건 사용자 평점
- **[USDA FoodData Central](https://fdc.nal.usda.gov/)** (`FoodData_Central_foundation_food_csv_2026-04-30.zip`)
  - 식재료별 정밀 영양성분 (단백질, 지방, 탄수화물, 당류, 식이섬유)

## 설치 및 실행

```bash
pip install -r requirements.txt
```

### 대화형 모드

```bash
python main.py
```

```
냉장고 재료를 쉼표로 입력하세요: chicken, garlic, broccoli, olive oil
사용자 ID (없으면 Enter): 
최소 당뇨 적합도 점수 (기본 30, 0-100): 50
```

### 비대화형 모드

```bash
python main.py --ingredients chicken garlic broccoli --top-n 5 --min-score 50
```

### 평가

```bash
python evaluate.py
```

## 당뇨 적합도 점수 기준

| 점수 | 등급 | 기준 |
|------|------|------|
| 75-100 | 매우 적합 | 당분 낮음 + 탄수화물 낮음 |
| 50-74 | 적합 | 당분/탄수화물 보통 이하 |
| 25-49 | 주의 필요 | 당분 또는 탄수화물 높음 |
| 0-24 | 비권장 | 당분·탄수화물 모두 높음 |

### 점수 산정 근거

[ADA Standards of Care 2025](https://diabetesjournals.org/care/article/48/Supplement_1/S86/157563/5-Facilitating-Positive-Health-Behaviors-and-Well)는 특정 %DV 수치를 고정하지 않고, 아래 방향성만 제시합니다.

- 탄수화물 섭취 줄이기 → 혈당 개선에 가장 효과적
- 첨가당·정제 곡물 최소화
- 식이섬유 최소 14g / 1,000kcal 권장
- 개인별 맞춤 식단 권장 (고정 수치 없음)

본 시스템의 임계값(sugar_pct, carbs_pct 기준선)은 FDA %DV 기준량(첨가당 DV=50g, 탄수화물 DV=275g)을 참고해 ADA의 방향성에 맞게 프로젝트 수준에서 설정한 **경험적 근사값**이며, ADA 공식 수치가 아닙니다.

## 프로젝트 구조

```
diabetes_ml/
├── src/
│   ├── data_loader.py      # 데이터 로딩 및 전처리
│   ├── diabetes_scorer.py  # 당뇨 적합도 점수 계산
│   ├── content_based.py    # TF-IDF 콘텐츠 기반 필터링
│   ├── collaborative.py    # SVD 협업 필터링
│   └── recommender.py      # 통합 추천 시스템
├── main.py                 # 실행 진입점
├── evaluate.py             # 모델 평가
└── requirements.txt
```
