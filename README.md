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

ADA(미국당뇨병학회) 식이 가이드라인 기준으로 산정.

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
