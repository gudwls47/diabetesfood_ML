"""
Blood glucose rise prediction model.

예측 방식: 순 탄수화물 비례 공식
  BG_rise = BASE_RISE + net_carbs_g * CARB_FACTOR  (최대 80 mg/dL)

배경
----
archive(2) CGM 데이터(141건) 분석 결과, 영양소와 혈당 상승의 상관관계가
거의 0(-0.09)으로 나타나 RandomForest가 평균값으로만 수렴함.
=> 대신 순 탄수화물에 비례하는 공식을 사용, 직관적이고 일관된 예측 제공.

계수 기준
---------
  BASE_RISE  = 8  mg/dL  : 탄수화물 외 기본 상승량 (archive(2) 최솟값 근거)
  CARB_FACTOR = 0.30      : 순탄수 100g -> +30 mg/dL 추가 상승
  상한 80 mg/dL           : 단일 식사 최대 혈당 상승 임상 한계치

주의: 개인차가 크므로 참고용으로만 활용하세요.
"""

BASE_RISE   = 8.0    # mg/dL
CARB_FACTOR = 0.30   # mg/dL per g net_carbs
MAX_RISE    = 80.0   # mg/dL

# 식사 유형별 보정 계수 (아침 인슐린 저항성, 저녁 활동량 감소 반영)
MEAL_FACTOR = {
    "breakfast": 1.1,
    "lunch":     1.0,
    "dinner":    0.9,
    "snacks":    1.05,
}


class BGRiseModel:
    """순 탄수화물 기반 공식으로 식후 혈당 상승(mg/dL)을 예측합니다."""

    def __init__(self):
        self._fitted = False

    def fit(self, *args, **kwargs) -> "BGRiseModel":
        """공식 기반 모델은 별도 학습이 필요 없습니다."""
        self._fitted = True
        return self

    def predict(
        self,
        nutrition: dict,
        meal_type: str = "lunch",
        pre_bg: float = 110.0,
    ) -> float:
        """
        식후 혈당 상승량(mg/dL)을 예측합니다.

        Parameters
        ----------
        nutrition : 'net_carbs_g' 키를 포함한 영양소 dict
        meal_type : 'breakfast' | 'lunch' | 'dinner' | 'snacks'
        pre_bg    : 식전 혈당 (공식에는 미사용, API 호환 유지)

        Returns
        -------
        예측 BG rise (mg/dL)
        """
        net_carbs = max(0.0, nutrition.get("net_carbs_g", 0.0))
        factor    = MEAL_FACTOR.get(meal_type.lower(), 1.0)
        rise      = (BASE_RISE + net_carbs * CARB_FACTOR) * factor
        return round(min(MAX_RISE, max(0.0, rise)), 1)

    def classify(self, bg_rise: float, pre_bg: float = 110.0) -> tuple[str, str, float]:
        """
        식후 2시간 혈당을 계산하여 대한당뇨병학회 기준으로 적합/부적합 판정.

        기준 (출처: 대한당뇨병학회 https://www.diabetes.or.kr)
          - 식전 혈당 목표: 80~130 mg/dL
          - 식후 2시간 혈당 목표: 180 mg/dL 미만

        판정 로직
          post_meal_bg = pre_bg + bg_rise
          < 180 mg/dL  -> 적합
          >= 180 mg/dL -> 부적합

        Returns
        -------
        (label, description, post_meal_bg)
        """
        post_meal_bg = round(pre_bg + bg_rise, 1)
        if post_meal_bg < 180:
            label = "적합"
            desc  = (f"식후 혈당 예측 {post_meal_bg:.0f} mg/dL "
                     f"(기준 180 미만 -- 목표 달성)")
        else:
            label = "부적합"
            desc  = (f"식후 혈당 예측 {post_meal_bg:.0f} mg/dL "
                     f"(기준 180 초과 -- {post_meal_bg - 180:.0f} mg/dL 초과)")
        return label, desc, post_meal_bg
