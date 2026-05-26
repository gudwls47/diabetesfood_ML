"""
Blood glucose rise lookup model.

archive(2).zip CGM + 식사 기록에서 음식별 실측 평균 BG rise를 추출해
레시피 이름 매칭으로 혈당 상승을 예측합니다.

방식
----
1. archive(2) food_data.csv 의 각 식사 기록에서:
   - 식전 혈당 = 식사 직전 CGM 측정값
   - 식후 혈당 = 식사 후 30분~2시간 내 최고 CGM 측정값
   - BG rise   = 식후 혈당 - 식전 혈당
2. 식사에 포함된 각 음식에 해당 BG rise를 연결
3. 음식별 평균 BG rise 딕셔너리 구성
4. 추천 레시피 이름과 매칭해 BG rise 반환

예시
----
  archive(2) 측정:
    "Veg Biryani (150 gm), Raita (100 gm)"  → +48 mg/dL
    "Veg Biryani (100 gm), Salad (50 gm)"   → +42 mg/dL
  => "veg biryani" 평균 BG rise = 45 mg/dL

주의: 식사에 여러 음식이 있으면 각 음식에 동일한 BG rise가 연결됩니다.
      개인차가 크므로 참고용으로만 활용하세요.
"""

import re
import zipfile
import pandas as pd


def _build_bg_rise_lookup(archive2_path: str) -> dict[str, float]:
    """
    archive(2) 실측 데이터에서 음식별 평균 BG rise(mg/dL) 딕셔너리를 생성합니다.

    Returns
    -------
    dict: {음식명(소문자): 평균 BG rise}
    """
    with zipfile.ZipFile(archive2_path) as z:
        with z.open("blood_sugar_data.csv") as f:
            bg = pd.read_csv(f)
        with z.open("food_data.csv") as f:
            food = pd.read_csv(f)

    bg["datetime"] = pd.to_datetime(
        bg["Date"] + " " + bg["Time"], format="%d-%m-%Y %I:%M %p"
    )
    food["datetime"] = pd.to_datetime(
        food["Date"] + " " + food["Time"], format="%d-%m-%Y %I:%M %p"
    )
    bg = bg.sort_values("datetime").reset_index(drop=True)

    food_rises: dict[str, list[float]] = {}

    for _, meal in food.iterrows():
        meal_time = meal["datetime"]

        # 식전 혈당: 식사 직전 마지막 CGM 측정값
        pre = bg[bg["datetime"] <= meal_time]["Blood Sugar Level"]
        if pre.empty:
            continue
        pre_bg = float(pre.iloc[-1])

        # 식후 혈당: 식사 후 30분~2시간 내 최고값
        post_mask = (
            (bg["datetime"] >= meal_time + pd.Timedelta(minutes=30)) &
            (bg["datetime"] <= meal_time + pd.Timedelta(hours=2))
        )
        post = bg[post_mask]["Blood Sugar Level"]
        if post.empty:
            continue

        bg_rise = float(post.max()) - pre_bg
        if bg_rise < 0:
            continue  # 노이즈 / 중복 식사 제거

        # 식사에 포함된 각 음식에 BG rise 연결
        for part in str(meal["Food Items"]).split(","):
            name = re.sub(r"\(.*?\)", "", part).strip().lower()
            if name:
                food_rises.setdefault(name, []).append(bg_rise)

    return {
        food: round(sum(rises) / len(rises), 1)
        for food, rises in food_rises.items()
    }


class BGRiseModel:
    """
    archive(2) 실측 데이터 기반 혈당 상승 예측 모델.

    레시피 이름 → archive(2) 음식명 매칭 → 실측 평균 BG rise 반환.
    """

    def __init__(self):
        self._lookup: dict[str, float] = {}  # {food_name: mean_bg_rise}
        self._fitted = False

    def fit(self, archive2_path: str = None, **kwargs) -> "BGRiseModel":
        """archive(2)에서 음식별 BG rise 데이터를 로딩합니다."""
        if archive2_path:
            self._lookup = _build_bg_rise_lookup(archive2_path)
        self._fitted = True
        return self

    def predict_by_name(self, recipe_name: str) -> float:
        """
        레시피 이름으로 archive(2) 실측 BG rise를 조회합니다.

        매칭 방식: archive(2) 음식명이 레시피 이름에 포함되거나 그 반대인 경우.
        미매칭 시 전체 평균값을 반환합니다.
        """
        name_lower = recipe_name.lower()
        for food, rise in self._lookup.items():
            if food in name_lower or name_lower in food:
                return rise

        # 매칭 실패 시 전체 평균
        if self._lookup:
            return round(sum(self._lookup.values()) / len(self._lookup), 1)
        return 20.0

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
