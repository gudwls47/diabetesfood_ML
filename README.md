# 당뇨 환자를 위한 냉장고 레시피 추천 시스템

냉장고 재료(한국어/영어)를 입력하면 만들 수 있는 레시피를 추천하고,
45명의 실제 CGM 데이터로 학습한 ML 모델이 예상 혈당 상승값과 당뇨 적합 여부를 알려줍니다.

---

## 실행 흐름

```
입력: "감자, 양파, 당근"
  ↓ 한국어 → 영어 변환 (potato, onion, carrot)
  ↓ Food.com 레시피 ~229,000개에서 재료 보유율 필터링
    (알코올·칵테일류 ~2,600개 자동 제거)
    (결과 없으면 자동 완화: 요청값 → 50 → 30 → 20 → 10%)
  ↓ 레시피 내장 영양성분 사용 (600 kcal 1인분 기준으로 정규화)
  ↓ Gradient Boosting 모델로 혈당 상승 예측
  ↓
[결과]
  레시피명       : Potato Carrot Soup
  재료 보유율    : 60%
  추가 필요 재료 : celery, chicken broth
  칼로리         : 320 kcal
  탄수화물       : 21.3g  |  단백질: 8.5g  |  지방: 6.2g
  혈당 상승 예측 : +35 mg/dL -> 식후 혈당 145 mg/dL
  당뇨 적합성    : [적합]  식후 혈당 예측 145 mg/dL (기준 180 미만 -- 목표 달성)
```

---

## 시스템 구성

### 파이프라인

| 단계 | 내용 | 데이터 출처 |
|------|------|------------|
| 1 | **레시피 필터링** | 알코올·칵테일류 제거 후 재료 보유율로 필터링 | archive.zip |
| 2 | **영양성분** | 레시피 내장 영양성분을 600 kcal 1인분 기준으로 정규화 | archive.zip |
| 3 | **혈당 상승 예측** | Gradient Boosting 회귀 모델 | CGMacros.zip |
| 4 | **당뇨 적합성 판정** | 식후 2시간 혈당 180 mg/dL 기준 (대한당뇨병학회) | - |

---

### 데이터 전처리

#### 1인분 정규화 (600 kcal 기준)

Food.com 영양성분은 레시피 전체(다인분) 기준입니다.
칼로리가 600 kcal를 초과하는 레시피는 비례 축소하여 1인분으로 정규화합니다.

```
ratio = min(1.0, 600 / 레시피_총칼로리)
carbs_g   = 원본_carbs   × ratio
protein_g = 원본_protein × ratio
fat_g     = 원본_fat     × ratio
calories  = 원본_calories × ratio
```

전체 레시피의 약 80%는 이미 600 kcal 이하이므로 변경 없음.

#### 부적절 레시피 필터

레시피명에 알코올 관련 단어(rum, gin, vodka, wine, beer, whiskey, martini, cocktail 등)가
**단어 경계(word boundary)** 기준으로 포함된 레시피를 자동 제거합니다.
(`ginger`, `crumb` 등 비슷한 철자의 정상 재료는 제거되지 않음)

---

### 혈당 예측 모델 — CGMacros 기반 Gradient Boosting

#### 학습 데이터: PhysioNet CGMacros

| 항목 | 내용 |
|------|------|
| 참여자 | 45명 (건강인 15 / 전당뇨 16 / **제2형 당뇨 14**) |
| 수집 기간 | 인당 약 10일 |
| CGM 기기 | Abbott FreeStyle Libre / Dexcom G6 (5~15분 간격) |
| 총 식사 기록 | **1,524건** |
| 데이터 출처 | [PhysioNet CGMacros v1.0.0](https://physionet.org/content/cgmacros/1.0.0/) |

각 식사 기록에서 다음과 같이 BG rise를 계산합니다.

```
식전 혈당  = 식사 직전 마지막 CGM 측정값
식후 혈당  = 식사 후 30분~2시간 내 최고 CGM 측정값
BG rise   = 식후 최고값 - 식전값
```

#### 모델 특성 (Features)

| 특성 | 중요도 | 설명 |
|------|--------|------|
| pre_bg | **0.343** | 식전 혈당 (mg/dL) — 가장 중요한 특성 |
| net_carbs_g | 0.143 | 순 탄수화물 = 탄수화물 - 식이섬유 (g) |
| carbs_g | 0.105 | 탄수화물 (g) |
| protein_g | 0.099 | 단백질 (g) |
| meal_type | 0.096 | 식사 유형 (breakfast / lunch / dinner) |
| calories | 0.084 | 열량 (kcal) |
| fat_g | 0.066 | 지방 (g) |
| fiber_g | 0.064 | 식이섬유 (g) |

> **Food.com 데이터에는 식이섬유 정보가 없어** `fiber_g = 0`으로 처리합니다.
> 식이섬유 중요도가 가장 낮은(0.064) 특성이므로 예측 영향은 미미합니다.

> **식전 혈당이 가장 중요한 이유**: 혈당이 이미 높은 상태에서 식사하면 식후 혈당이 더 크게 오릅니다.
> 이 패턴이 45명의 실측 데이터에서 학습되었습니다.

---

### 당뇨 적합성 판정

```
식후 혈당 예측 = 식전 혈당 + 혈당 상승 예측값
```

| 등급 | 조건 | 설명 |
|------|------|------|
| **적합** | 식후 혈당 < 180 mg/dL | 목표 달성 |
| **부적합** | 식후 혈당 >= 180 mg/dL | 목표 초과 |

**출처: [대한당뇨병학회](https://www.diabetes.or.kr/general/info/treat/treat_01.php)**
- 식전 혈당 목표: 80~130 mg/dL
- 식후 2시간 혈당 목표: **180 mg/dL 미만**

---

## 데이터셋

| 파일 | 출처 | 용도 |
|------|------|------|
| `archive.zip` | [Food.com Recipes Dataset (Kaggle)](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) | 레시피 231,637개 + 내장 영양성분 |
| `CGMacros_dateshifted365.zip` | [PhysioNet CGMacros v1.0.0](https://physionet.org/content/cgmacros/1.0.0/) | 45명 CGM + 식사 기록 — ML 모델 학습 |

> **Food.com 영양성분**: calories, carbs, fat, protein (%DV → g 변환, 600 kcal 1인분 정규화).
> 식이섬유 데이터 없음 → `fiber_g = 0` 고정.

---

## 설치 및 실행

```bash
pip install -r requirements.txt
```

### 데이터 파일 경로 설정

`main.py` 상단의 경로를 본인 환경에 맞게 수정하세요.

```python
DEFAULT_ARCHIVE_FOOD = r"경로\archive.zip"
DEFAULT_CGMACROS     = r"경로\CGMacros_dateshifted365.zip"
```

### 대화형 모드

```bash
python main.py
```

```
식전 혈당 입력 (mg/dL, 80~130) 또는 'q' 종료: 118
냉장고 재료 입력 (쉼표 구분): 감자, 양파, 당근
식사 유형 (breakfast/lunch/dinner, 기본 lunch): lunch
최소 재료 보유율 % (기본 50): 50
```

한국어, 영어 모두 입력 가능합니다. (`감자` = `potato`)

### 비대화형 모드

```bash
python main.py --ingredients 감자 양파 당근 --pre-bg 118 --meal-type lunch --top-n 5
python main.py --ingredients chicken garlic broccoli --pre-bg 95 --min-coverage 0.4
```

당뇨/저탄수화물 태그 레시피만 빠르게 로드하려면:

```bash
python main.py --diabetic-only --ingredients chicken broccoli --pre-bg 110
```

### 모델 평가

```bash
python evaluate.py
```

CGMacros 데이터 통계, 5-fold 교차검증 RMSE, 특성 중요도, 레시피 적합성 분포를 출력합니다.

---

## 프로젝트 구조

```
diabetes_ml/
├── src/
│   ├── cgmacros_loader.py    # CGMacros 45명 CSV 파싱, 식사별 BG rise 추출
│   ├── data_loader.py        # Food.com 레시피 로딩 (영양성분 정규화, 알코올 필터)
│   ├── translator.py         # 한국어 재료명 -> 영어 변환 딕셔너리
│   ├── ingredient_matcher.py # 재료 보유율 계산 및 레시피 필터링 (자동 임계값 완화)
│   ├── bg_model.py           # Gradient Boosting BG rise 예측 모델
│   └── recommender.py        # 전체 파이프라인 통합
├── scripts/
│   ├── analyze_archive.py         # Food.com 데이터 통계 분석
│   ├── find_high_bg.py            # BG rise 50+ 레시피 탐색
│   ├── find_high_bg_ingredients.py# 고위험 레시피 공통 재료 분석
│   ├── show_filtered.py           # 알코올 필터 적용 결과 확인
│   └── test_serving.py            # 1인분 정규화 결과 검증
├── main.py                   # 실행 진입점 (대화형 / CLI)
├── evaluate.py               # 모델 평가 (CV RMSE, 특성 중요도, 적합성 분포)
└── requirements.txt
```
