# 당뇨 환자를 위한 냉장고 레시피 추천 시스템

냉장고 재료(한국어/영어)를 입력하면 만들 수 있는 레시피를 추천하고,
실제 CGM 데이터로 학습한 ML 모델이 예상 혈당 상승값과 당뇨 적합 여부를 알려줍니다.

---

## 실행 흐름

```
입력: "감자, 양파, 당근"
  ↓ 한국어 → 영어 변환 (potato, onion, carrot)
  ↓ Food.com 레시피에서 재료 보유율 50% 이상 필터링
  ↓ 식품 영양성분 DB에서 탄수화물·식이섬유·단백질 계산
  ↓ 혈당 상승 예측 (Random Forest, 실측 CGM 데이터 학습)
  ↓
[결과]
  레시피명       : 30 minute baked potato
  재료 보유율    : 50%
  순 탄수화물    : 44.4g  (탄수화물 - 식이섬유)
  혈당 상승 예측 : +12 mg/dL
  당뇨 적합성    : [적합]
```

---

## ML 모델 구성

| 단계 | 방식 | 데이터 출처 |
|------|------|------------|
| 1 | **재료 매칭** | 서브스트링 매칭으로 보유 재료로 만들 수 있는 레시피 필터링 | Food.com |
| 2 | **영양성분 계산** | 재료별 탄수화물·식이섬유·당류·단백질·지방 합산 | 식품 영양성분 DB |
| 3 | **혈당 상승 예측** | Random Forest 회귀 모델 | CGM + 식사 일지 |
| 4 | **협업 필터링** | Truncated SVD (Matrix Factorization) 개인화 | Food.com 사용자 평점 |

### 혈당 예측 모델 상세

- **학습 데이터**: 1인의 76일 CGM 측정값(761건) + 식사 일지(247건)
- **피처**: 순 탄수화물(g), 당류(g), 식이섬유(g), 단백질(g), 지방(g), 칼로리, 식사 유형
- **타겟**: 식사 후 30분~2시간 내 최고 혈당 - 식전 혈당 (mg/dL)
- **모델**: RandomForestRegressor
- **CV RMSE**: 약 8 mg/dL

> **주의**: 단일 피험자 데이터 기반 추정치입니다. 개인차가 크므로 임상 판단 대체 불가.

### 당뇨 적합성 판정 기준

```
식후 혈당 예측 = 식전 혈당 + 혈당 상승 예측값
```

| 등급 | 조건 | 설명 |
|------|------|------|
| **적합** | 식후 혈당 예측 < 180 mg/dL | 목표 달성 |
| **부적합** | 식후 혈당 예측 >= 180 mg/dL | 목표 초과 |

**출처: [대한당뇨병학회](https://www.diabetes.or.kr/general/info/treat/treat_01.php)**
- 식전 혈당 목표: 80~130 mg/dL
- 식후 2시간 혈당 목표: **180 mg/dL 미만**
- 당화혈색소 목표: 6.5% 미만

---

## 데이터셋

| 파일 | 출처 | 용도 |
|------|------|------|
| `archive.zip` | [Food.com Recipes and Interactions (Kaggle)](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) | 레시피 23만 개 + 사용자 평점 |
| `archive (1).zip` | [Food Nutrition Dataset (Kaggle)](https://www.kaggle.com/datasets/utsavdey1410/food-nutrition-dataset) | 2,395개 식품 영양성분 (실제 g값) |
| `archive (2).zip` | [Food consumed and corresponding blood sugar change (Kaggle)](https://www.kaggle.com/datasets/suyashmaurya/food-consumed-and-corresponding-blood-sugar-change) | 실측 혈당 상승값 → 모델 학습 |

---

## 설치 및 실행

```bash
pip install -r requirements.txt
```

### 데이터 파일 경로 설정

`main.py` 상단의 경로를 본인 환경에 맞게 수정하세요.

```python
DEFAULT_ARCHIVE   = r"경로\archive.zip"
DEFAULT_FOODDATA1 = r"경로\archive (1).zip"
DEFAULT_ARCHIVE2  = r"경로\archive (2).zip"
```

### 대화형 모드

```bash
python main.py
```

```
식전 혈당 입력 (mg/dL, 80~130) 또는 'q' 종료: 118
냉장고 재료 입력 (쉼표 구분): 감자, 양파, 당근
식사 유형 (breakfast/lunch/dinner/snacks, 기본 lunch): lunch
최소 재료 보유율 % (기본 50): 50
```

한국어, 영어 모두 입력 가능합니다. (`감자` = `potato`)

### 비대화형 모드

```bash
python main.py --ingredients 감자 양파 당근 --pre-bg 118 --meal-type lunch --top-n 5
python main.py --ingredients chicken garlic broccoli --pre-bg 95 --min-coverage 0.4
```

### 모델 평가

```bash
python evaluate.py
```

---

## 프로젝트 구조

```
diabetes_ml/
├── src/
│   ├── data_loader.py        # Food.com 레시피·평점 데이터 로딩
│   ├── translator.py         # 한국어 재료명 → 영어 변환 딕셔너리
│   ├── ingredient_matcher.py # 재료 보유율 계산 및 레시피 필터링
│   ├── nutrition_db.py       # 식품 영양성분 DB 로딩 및 재료 매칭
│   ├── bg_model.py           # CGM 데이터 기반 혈당 상승 회귀 모델
│   ├── collaborative.py      # SVD 협업 필터링 (개인화)
│   └── recommender.py        # 전체 파이프라인 통합
├── main.py                   # 실행 진입점 (대화형 / CLI)
├── evaluate.py               # 모델 평가 (RMSE, CB 정확도)
└── requirements.txt
```
