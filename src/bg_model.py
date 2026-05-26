"""
Blood glucose rise prediction model — CGMacros 기반 Random Forest.

데이터: PhysioNet CGMacros (45명, 건강인 15 / 전당뇨 16 / 제2형 당뇨 14)
       각 참여자의 실측 CGM + 식사 영양성분 데이터

학습 특성 (Features)
--------------------
  carbs_g      : 탄수화물 (g)
  protein_g    : 단백질 (g)
  fat_g        : 지방 (g)
  fiber_g      : 식이섬유 (g)
  net_carbs_g  : 순 탄수화물 = carbs - fiber (g)
  calories     : 열량 (kcal)
  meal_type    : 0=breakfast / 1=lunch / 2=dinner
  pre_bg       : 식전 혈당 (mg/dL)

예측 대상 (Target)
------------------
  bg_rise : 식후 최고 혈당 - 식전 혈당 (mg/dL)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from .cgmacros_loader import load_cgmacros

_MEAL_ENC = {"breakfast": 0, "lunch": 1, "dinner": 2}


def _to_features(
    carbs_g: float,
    protein_g: float,
    fat_g: float,
    fiber_g: float,
    net_carbs_g: float,
    calories: float,
    meal_type: str,
    pre_bg: float,
) -> list[float]:
    meal_enc = _MEAL_ENC.get(str(meal_type).lower(), 1)
    return [carbs_g, protein_g, fat_g, fiber_g, net_carbs_g, calories, meal_enc, pre_bg]


class BGRiseModel:
    """
    CGMacros 실측 데이터 기반 혈당 상승 예측 모델.

    45명(당뇨 포함)의 CGM + 식사 기록으로 학습한 Gradient Boosting Regressor.
    영양성분(탄수화물·단백질·지방·식이섬유)과 식전 혈당으로 BG rise를 예측합니다.
    """

    FEATURE_NAMES = [
        "carbs_g", "protein_g", "fat_g", "fiber_g",
        "net_carbs_g", "calories", "meal_type", "pre_bg",
    ]

    def __init__(self):
        self._model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        self._fitted = False
        self._mean_bg_rise = 20.0  # fallback
        self._n_samples = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, cgmacros_path: str, **kwargs) -> "BGRiseModel":
        """
        CGMacros zip에서 데이터를 로드하고 모델을 학습합니다.

        Parameters
        ----------
        cgmacros_path : CGMacros_dateshifted365.zip 경로
        """
        df = load_cgmacros(cgmacros_path)
        df = self._clean(df)
        self._n_samples = len(df)
        self._mean_bg_rise = round(float(df["bg_rise"].mean()), 1)

        X = df[self.FEATURE_NAMES].values
        y = df["bg_rise"].values

        self._model.fit(X, y)
        self._fitted = True
        return self

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # meal_type 인코딩
        df["meal_type"] = df["meal_type"].map(_MEAL_ENC).fillna(1).astype(int)
        # 이상값 제거
        df = df[df["bg_rise"] >= 0]
        df = df[df["bg_rise"] <= 150]       # 150 mg/dL 초과는 제거
        df = df[df["carbs_g"] <= 300]
        df = df.dropna(subset=BGRiseModel.FEATURE_NAMES + ["bg_rise"])
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        carbs_g: float,
        protein_g: float,
        fat_g: float,
        fiber_g: float,
        net_carbs_g: float,
        calories: float,
        meal_type: str = "lunch",
        pre_bg: float = 110.0,
    ) -> float:
        """
        영양성분과 식전 혈당으로 BG rise를 예측합니다.

        Returns
        -------
        float : 예측 BG rise (mg/dL)
        """
        if not self._fitted:
            return self._mean_bg_rise

        x = np.array([_to_features(
            carbs_g, protein_g, fat_g, fiber_g,
            net_carbs_g, calories, meal_type, pre_bg,
        )]).reshape(1, -1)

        rise = float(self._model.predict(x)[0])
        return round(max(0.0, rise), 1)

    def predict_from_nutrition(
        self,
        nutr: dict,
        meal_type: str = "lunch",
        pre_bg: float = 110.0,
    ) -> float:
        """
        estimate_recipe_nutrition() 반환값 dict로 BG rise를 예측합니다.

        Parameters
        ----------
        nutr      : {'carbs_g', 'protein_g', 'fat_g', 'fiber_g', 'net_carbs_g', 'calories', ...}
        meal_type : 'breakfast' | 'lunch' | 'dinner'
        pre_bg    : 식전 혈당 (mg/dL)
        """
        return self.predict(
            carbs_g     = float(nutr.get("carbs_g",     0) or 0),
            protein_g   = float(nutr.get("protein_g",   0) or 0),
            fat_g       = float(nutr.get("fat_g",       0) or 0),
            fiber_g     = float(nutr.get("fiber_g",     0) or 0),
            net_carbs_g = float(nutr.get("net_carbs_g", 0) or 0),
            calories    = float(nutr.get("calories",    0) or 0),
            meal_type   = meal_type,
            pre_bg      = pre_bg,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def cv_rmse(self, cgmacros_path: str, cv: int = 5) -> tuple[float, float]:
        """
        5-fold 교차검증 RMSE를 반환합니다 (fit() 전에 호출 가능).

        Returns
        -------
        (mean_rmse, std_rmse)
        """
        df = load_cgmacros(cgmacros_path)
        df = self._clean(df)
        X = df[self.FEATURE_NAMES].values
        y = df["bg_rise"].values

        scores = cross_val_score(
            self._model, X, y,
            cv=cv, scoring="neg_root_mean_squared_error",
        )
        rmse_scores = -scores
        return round(float(rmse_scores.mean()), 2), round(float(rmse_scores.std()), 2)

    def feature_importances(self) -> pd.Series:
        """학습된 모델의 특성 중요도를 반환합니다."""
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        return pd.Series(
            self._model.feature_importances_,
            index=self.FEATURE_NAMES,
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        bg_rise: float,
        pre_bg: float = 110.0,
    ) -> tuple[str, str, float]:
        """
        식후 2시간 혈당을 계산하여 대한당뇨병학회 기준으로 적합/부적합 판정.

        기준 (출처: 대한당뇨병학회 https://www.diabetes.or.kr)
          - 식후 2시간 혈당 목표: 180 mg/dL 미만
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
