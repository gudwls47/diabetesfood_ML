"""
Blood glucose rise lookup model.

archive(2)와 archive(5)를 계층적으로 결합하여 혈당 상승을 예측합니다.

우선순위
-------
1. archive(2) CGM 실측값  : 레시피명 매칭 -> 실측 평균 BG rise (mg/dL)
2. archive(5) GI 추정값   : 레시피명 매칭 -> GI 구간별 BG rise 추정 (mg/dL)
3. 전체 평균 fallback      : 매칭 실패 시 archive(2) 전체 평균 사용

archive(2) 방식
--------------
1. food_data.csv 각 식사 기록에서:
   - 식전 혈당 = 식사 직전 CGM 측정값
   - 식후 혈당 = 식사 후 30분~2시간 내 최고 CGM 측정값
   - BG rise   = 식후 혈당 - 식전 혈당
2. 음식별 평균 BG rise 딕셔너리 구성

archive(5) GI 방식
------------------
1. pred_food.csv 에서 음식별 평균 GI 로드
2. GI 구간으로 BG rise 추정:
   GI < 55  -> 15 mg/dL  (낮음)
   GI 55-69 -> 25 mg/dL  (중간)
   GI >= 70 -> 40 mg/dL  (높음)
"""

import re
import zipfile
import pandas as pd


def _build_bg_rise_lookup(archive2_path: str) -> dict[str, float]:
    """
    archive(2) 실측 데이터에서 음식별 평균 BG rise(mg/dL) 딕셔너리를 생성합니다.
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


def _build_gi_lookup(archive5_path: str) -> dict[str, float]:
    """
    archive(5) pred_food.csv에서 음식별 평균 GI 딕셔너리를 생성합니다.
    중복 음식명은 GI 평균으로 처리합니다.
    """
    with zipfile.ZipFile(archive5_path) as z:
        with z.open("pred_food.csv") as f:
            df = pd.read_csv(f)

    gi_dict: dict[str, float] = {}
    for name, grp in df.groupby("Food Name"):
        gi_dict[name.lower()] = round(grp["Glycemic Index"].mean(), 1)
    return gi_dict


def _gi_to_bg_rise(gi: float) -> float:
    """GI 구간 -> 혈당 상승 추정값 (mg/dL)"""
    if gi < 55:
        return 15.0   # 낮은 GI
    elif gi < 70:
        return 25.0   # 중간 GI
    else:
        return 40.0   # 높은 GI


class BGRiseModel:
    """
    archive(2) CGM 실측 + archive(5) GI 계층적 혈당 상승 예측 모델.

    레시피 이름 매칭 순서:
      1순위 archive(2) -> 실측 평균 BG rise (mg/dL)
      2순위 archive(5) -> GI 구간별 BG rise 추정 (mg/dL)
      3순위 전체 평균  -> archive(2) 전체 평균 fallback
    """

    def __init__(self):
        self._lookup: dict[str, float]    = {}   # {food_name: mean_bg_rise}
        self._gi_lookup: dict[str, float] = {}   # {food_name: mean_gi}
        self._fitted = False

    def fit(
        self,
        archive2_path: str = None,
        archive5_path: str = None,
        **kwargs,
    ) -> "BGRiseModel":
        """archive(2) 실측 데이터와 archive(5) GI 데이터를 로딩합니다."""
        if archive2_path:
            self._lookup = _build_bg_rise_lookup(archive2_path)
        if archive5_path:
            self._gi_lookup = _build_gi_lookup(archive5_path)
        self._fitted = True
        return self

    def predict_by_name(self, recipe_name: str) -> tuple[float, str]:
        """
        레시피 이름으로 BG rise를 조회합니다.

        Returns
        -------
        (bg_rise_mg_dl, source)
          'cgm'  - archive(2) CGM 실측값
          'gi'   - archive(5) GI 기반 추정값
          'mean' - archive(2) 전체 평균 (매칭 실패)
        """
        name_lower = recipe_name.lower()

        # 1순위: archive(2) CGM 실측값
        for food, rise in self._lookup.items():
            if food in name_lower or name_lower in food:
                return rise, "cgm"

        # 2순위: archive(5) GI 추정값
        for food, gi in self._gi_lookup.items():
            if food in name_lower or name_lower in food:
                return _gi_to_bg_rise(gi), "gi"

        # 3순위: 전체 평균 fallback
        if self._lookup:
            return round(sum(self._lookup.values()) / len(self._lookup), 1), "mean"
        return 20.0, "mean"

    def classify(
        self,
        bg_rise: float,
        pre_bg: float = 110.0,
        source: str = "cgm",
    ) -> tuple[str, str, float]:
        """
        식후 2시간 혈당을 계산하여 대한당뇨병학회 기준으로 적합/부적합 판정.

        기준 (출처: 대한당뇨병학회 https://www.diabetes.or.kr)
          - 식전 혈당 목표: 80~130 mg/dL
          - 식후 2시간 혈당 목표: 180 mg/dL 미만
        """
        post_meal_bg = round(pre_bg + bg_rise, 1)
        src_label = {
            "cgm":  "CGM 실측",
            "gi":   "GI 추정",
            "mean": "평균 추정",
        }.get(source, source)

        if post_meal_bg < 180:
            label = "적합"
            desc  = (f"식후 혈당 예측 {post_meal_bg:.0f} mg/dL "
                     f"(기준 180 미만 -- 목표 달성) [{src_label}]")
        else:
            label = "부적합"
            desc  = (f"식후 혈당 예측 {post_meal_bg:.0f} mg/dL "
                     f"(기준 180 초과 -- {post_meal_bg - 180:.0f} mg/dL 초과) [{src_label}]")
        return label, desc, post_meal_bg
