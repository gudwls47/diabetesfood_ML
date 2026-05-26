# 당뇨 환자를 위한 냉장고 레시피 추천 시스템

냉장고 재료(한국어/영어)를 입력하면 만들 수 있는 레시피를 추천하고,
실제 CGM 데이터로 학습한 ML 모델이 예상 혈당 상승값과 당뇨 적합 여부를 알려줍니다.

---

## 실행 흐름

```
입력: "감자, 양파, 당근"
  ↓ 한국어 → 영어 변환 (potato, onion, carrot)
  ↓ 인도 레시피(Cleaned Indian Food Dataset, 2,825개)에서
    재료 보유율 필터링 (기본 50%, 결과 없으면 자동 완화: 50→30→20→10%)
  ↓ 식품 영양성분 DB에서 탄수화물·식이섬유·단백질 계산
  ↓ 혈당 상승 예측 (Random Forest, 실측 CGM 데이터 학습)
  ↓
[결과]
  레시피명       : Aloo Gobi (감자 & 콜리플라워 볶음)
  재료 보유율    : 60%
  순 탄수화물    : 21.3g  (탄수화물 - 식이섬유)
  혈당 상승 예측 : +18 mg/dL
  당뇨 적합성    : [적합]  식후 혈당 예측 128 mg/dL (기준 180 미만 - 목표 달성)
```

---

## ML 모델 구성

| 단계 | 방식 | 데이터 출처 |
|------|------|------------|
| 0 | **CGM 음식 필터링** | archive(2) 실측 음식명과 일치하는 레시피만 선별 (5,938 → 2,825개) | archive (2).zip |
| 1 | **재료 매칭** | 서브스트링 매칭으로 만들 수 있는 레시피 필터링 (결과 부족 시 임계값 자동 완화) | archive (3).zip |
| 2 | **영양성분 계산** | 재료별 탄수화물·식이섬유·당류·단백질·지방 합산 | archive (1).zip |
| 3 | **혈당 상승 예측** | Random Forest 회귀 모델 | archive (2).zip |

### 혈당 예측 방식

```
BG_rise = 8 + net_carbs_g × 0.30  (상한 80 mg/dL)
```

- **순 탄수화물 비례 공식** — 탄수화물이 높을수록 혈당이 더 오름
- 식사 유형 보정: 아침 ×1.1 / 점심 ×1.0 / 저녁 ×0.9 / 간식 ×1.05
- archive(2) CGM 데이터(141건) 분석 시 영양소-혈당 상관관계가 거의 0(-0.09)으로
  RandomForest가 평균값으로만 수렴하여, 직관적이고 일관된 공식 방식으로 전환

> **주의**: 개인차가 크므로 참고용으로만 활용하세요.

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
| `archive (1).zip` | [Food Nutrition Dataset (Kaggle)](https://www.kaggle.com/datasets/utsavdey1410/food-nutrition-dataset) | 2,395개 식품 영양성분 (실제 g값) |
| `archive (2).zip` | [Food consumed and corresponding blood sugar change (Kaggle)](https://www.kaggle.com/datasets/suyashmaurya/food-consumed-and-corresponding-blood-sugar-change) | 실측 혈당 상승값 → BG 모델 학습 |
| `archive (3).zip` | [Cleaned Indian Recipes Dataset (Kaggle)](https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset) | 인도 레시피 5,938개 → 레시피 추천 |

> **인도 레시피를 사용하는 이유**: 혈당 학습 데이터(`archive (2).zip`)가 인도 음식 기반이므로
> 같은 문화권 레시피를 사용하면 영양성분 분포가 일치해 혈당 예측 정확도가 높아집니다.

---

## 설치 및 실행

```bash
pip install -r requirements.txt
```

### 데이터 파일 경로 설정

`main.py` 상단의 경로를 본인 환경에 맞게 수정하세요.

```python
DEFAULT_FOODDATA1 = r"경로\archive (1).zip"
DEFAULT_ARCHIVE2  = r"경로\archive (2).zip"
DEFAULT_ARCHIVE3  = r"경로\archive (3).zip"
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
│   ├── data_loader.py        # 인도 레시피 로딩 및 CGM 음식명 기준 필터링
│   ├── translator.py         # 한국어 재료명 → 영어 변환 딕셔너리
│   ├── ingredient_matcher.py # 재료 보유율 계산 및 레시피 필터링 (자동 임계값 완화)
│   ├── nutrition_db.py       # 식품 영양성분 DB 로딩 및 재료별 영양 추정
│   ├── bg_model.py           # CGM 데이터 기반 혈당 상승 회귀 모델 (RandomForest)
│   └── recommender.py        # 전체 파이프라인 통합
├── main.py                   # 실행 진입점 (대화형 / CLI)
├── evaluate.py               # BG 모델 평가 (CV RMSE, 적합성 분포)
└── requirements.txt
```
