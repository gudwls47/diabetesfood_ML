# 당뇨 환자를 위한 냉장고 레시피 추천 시스템

냉장고 재료(한국어/영어)를 입력하면 만들 수 있는 인도 레시피를 추천하고,
실제 CGM 측정 데이터와 혈당지수(GI) DB를 결합해 예상 혈당 상승값과 당뇨 적합 여부를 알려줍니다.

---

## 실행 흐름

```
입력: "감자, 양파, 당근"
  ↓ 한국어 → 영어 변환 (potato, onion, carrot)
  ↓ 인도 레시피 3,952개에서 재료 보유율 필터링
    (기본 50%, 결과 없으면 자동 완화: 50→30→20→10%)
  ↓ 식품 영양성분 DB에서 탄수화물·식이섬유·단백질 계산
  ↓ 혈당 상승 예측 (3단계 계층 조회)
  ↓
[결과]
  레시피명       : Aloo Gobi
  재료 보유율    : 60%
  순 탄수화물    : 21.3g
  혈당 상승 예측 : +18 mg/dL  [CGM 실측]
  당뇨 적합성    : [적합]  식후 혈당 예측 128 mg/dL (기준 180 미만 -- 목표 달성) [CGM 실측]
```

---

## 시스템 구성

### 파이프라인

| 단계 | 내용 | 데이터 출처 |
|------|------|------------|
| 0 | **레시피 풀 구성** | archive(2) OR archive(5) 음식명 매칭 레시피 선별 (5,938 → 3,952개) | archive (2)(5).zip |
| 1 | **재료 매칭** | 서브스트링 매칭으로 만들 수 있는 레시피 필터링 | archive (3).zip |
| 2 | **영양성분 계산** | 재료별 탄수화물·식이섬유·당류·단백질·지방 합산 | archive (1).zip |
| 3 | **혈당 상승 예측** | 3단계 계층 조회 (아래 참고) | archive (2)(5).zip |
| 4 | **당뇨 적합성 판정** | 식후 2시간 혈당 180 mg/dL 기준 (대한당뇨병학회) | - |

---

### 혈당 예측 방식 — 3단계 계층 조회

레시피 이름을 기준으로 아래 순서로 조회하며, 앞 단계에서 매칭되면 즉시 반환합니다.

```
레시피명 매칭 시도
  ├─ 1순위: archive(2) CGM 실측값  ──→ 실제 측정 평균 BG rise (mg/dL)
  ├─ 2순위: archive(5) GI 추정값   ──→ GI 구간별 BG rise 추정 (mg/dL)
  └─ 3순위: 전체 평균 fallback     ──→ archive(2) 전체 평균 사용
```

#### 1순위: archive(2) CGM 실측값 (213개 음식)

archive(2)의 `blood_sugar_data.csv`(CGM 측정값)와 `food_data.csv`(식사 기록)를 결합해
음식별 평균 BG rise 딕셔너리를 구축합니다.

- **식전 혈당**: 식사 직전 마지막 CGM 측정값
- **식후 혈당**: 식사 후 30분~2시간 내 최고 CGM 측정값
- **BG rise**: 식후 최고값 - 식전값, 음식별 평균 계산

| 음식 | 실측 평균 BG rise |
|------|-----------------|
| chawal (쌀밥) | +41.5 mg/dL |
| dosa          | +40.0 mg/dL |
| chana sabji   | +38.0 mg/dL |
| veg biryani   | +29.0 mg/dL |
| roti          | +22.9 mg/dL |

#### 2순위: archive(5) GI 추정값 (253개 음식)

archive(5)의 `pred_food.csv`에서 음식별 혈당지수(GI)를 로드하고,
GI 구간으로 BG rise를 추정합니다.

| GI 구간 | 해당 음식 예 | 추정 BG rise |
|---------|------------|-------------|
| GI < 55 (낮음) | palak paneer, dal, raita | +15 mg/dL |
| GI 55-69 (중간) | veg biryani, pulao, dosa | +25 mg/dL |
| GI >= 70 (높음) | aloo paratha, puri, rice | +40 mg/dL |

#### 3순위: 전체 평균 fallback

1·2순위 모두 매칭 실패 시 archive(2) 전체 평균값을 사용합니다.

#### 실제 커버리지 (3,952개 레시피 기준)

| 출처 | 레시피 수 | 비율 |
|------|----------|------|
| [CGM 실측] archive(2) | 2,812개 | 71.2% |
| [GI 추정]  archive(5) | 1,135개 | 28.7% |
| [평균 추정] fallback  | 5개     | 0.1%  |

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
| `archive (1).zip` | [Food Nutrition Dataset (Kaggle)](https://www.kaggle.com/datasets/utsavdey1410/food-nutrition-dataset) | 2,395개 식품 영양성분 |
| `archive (2).zip` | [Food consumed and corresponding blood sugar change (Kaggle)](https://www.kaggle.com/datasets/suyashmaurya/food-consumed-and-corresponding-blood-sugar-change) | CGM 실측 혈당 데이터 (213개 음식) |
| `archive (3).zip` | [Cleaned Indian Recipes Dataset (Kaggle)](https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset) | 인도 레시피 5,938개 |
| `archive (5).zip` | [Diabetes Food Dataset (Kaggle)](https://www.kaggle.com/) | 혈당지수(GI) DB — 253개 음식 |

> **인도 레시피를 사용하는 이유**: archive(2) CGM 데이터가 인도 음식 기반이므로
> 같은 문화권 레시피를 사용하면 음식명 매칭률이 높고 혈당 예측 신뢰도가 올라갑니다.

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
DEFAULT_ARCHIVE5  = r"경로\archive (5).zip"
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

CGM 실측 통계, GI DB 통계, 레시피 적합성 분포, 예측 출처별 커버리지를 출력합니다.

---

## 프로젝트 구조

```
diabetes_ml/
├── src/
│   ├── data_loader.py        # 인도 레시피 로딩, archive(2)+(5) 음식명 기준 필터링
│   ├── translator.py         # 한국어 재료명 -> 영어 변환 딕셔너리
│   ├── ingredient_matcher.py # 재료 보유율 계산 및 레시피 필터링 (자동 임계값 완화)
│   ├── nutrition_db.py       # 식품 영양성분 DB 로딩 및 재료별 영양 추정
│   ├── bg_model.py           # CGM 실측 + GI 계층적 혈당 상승 예측
│   └── recommender.py        # 전체 파이프라인 통합
├── main.py                   # 실행 진입점 (대화형 / CLI)
├── evaluate.py               # 데이터 통계 및 레시피 적합성 분포 평가
└── requirements.txt
```
