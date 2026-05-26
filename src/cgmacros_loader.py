"""
CGMacros 데이터셋 로더 및 전처리.

PhysioNet CGMacros (https://physionet.org/content/cgmacros/1.0.0/)
  - 45명 참여자 (건강인 15 / 전당뇨 16 / 제2형 당뇨 14)
  - 인당 약 10일, CGM + 식사 기록

각 참여자 CSV 구조
------------------
  Timestamp     : 측정 시각 (1분 간격)
  Libre GL      : CGM 혈당 (mg/dL) — Abbott FreeStyle
  Dexcom GL     : CGM 혈당 (mg/dL) — Dexcom G6
  Meal Type     : 'Breakfast' | 'Lunch' | 'Dinner'  (식사 시작 행에만 값)
  Calories      : 식사 열량 (kcal)
  Carbs         : 탄수화물 (g)
  Protein       : 단백질 (g)
  Fat           : 지방 (g)
  Fiber         : 식이섬유 (g)
  Amount Consumed: 섭취 비율 (%)

처리 방식
---------
  각 식사 행에 대해:
  1. 식전 혈당 = 식사 직전 마지막 CGM 측정값
  2. 식후 최고 혈당 = 식사 후 30분~2시간 내 최고 CGM 측정값
  3. BG rise = 식후 최고 - 식전  (음수면 노이즈로 제거)
"""

import io
import re
import zipfile

import pandas as pd


def _pick_gl(df: pd.DataFrame) -> pd.Series:
    """
    Libre GL / Dexcom GL 중 유효한 쪽을 반환합니다.
    둘 다 있으면 평균을 사용합니다.
    """
    libre  = pd.to_numeric(df.get("Libre GL",  pd.Series(dtype=float)), errors="coerce")
    dexcom = pd.to_numeric(df.get("Dexcom GL", pd.Series(dtype=float)), errors="coerce")

    valid_libre  = libre.notna().sum()
    valid_dexcom = dexcom.notna().sum()

    if valid_libre > 0 and valid_dexcom > 0:
        return libre.combine_first(dexcom)
    elif valid_libre > 0:
        return libre
    else:
        return dexcom


def _extract_meals_from_df(df: pd.DataFrame) -> list[dict]:
    """
    단일 참여자 DataFrame에서 식사별 BG rise 레코드를 추출합니다.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["gl"] = _pick_gl(df)

    records = []

    meal_rows = df[df["Meal Type"].notna()].copy()
    for _, meal in meal_rows.iterrows():
        meal_time = meal["datetime"]

        # 식전 혈당: 식사 직전 마지막 유효 CGM
        pre_mask = df["datetime"] <= meal_time
        pre_gl   = df.loc[pre_mask, "gl"].dropna()
        if pre_gl.empty:
            continue
        pre_bg = float(pre_gl.iloc[-1])

        # 식후 혈당: 식사 후 30분~2시간 내 최고값
        post_mask = (
            (df["datetime"] >= meal_time + pd.Timedelta(minutes=30)) &
            (df["datetime"] <= meal_time + pd.Timedelta(hours=2))
        )
        post_gl = df.loc[post_mask, "gl"].dropna()
        if post_gl.empty:
            continue

        bg_rise = float(post_gl.max()) - pre_bg
        if bg_rise < 0:
            continue  # 노이즈 제거

        # 영양성분
        try:
            carbs   = float(meal.get("Carbs",   0) or 0)
            protein = float(meal.get("Protein", 0) or 0)
            fat     = float(meal.get("Fat",     0) or 0)
            fiber   = float(meal.get("Fiber",   0) or 0)
            cal     = float(meal.get("Calories", 0) or 0)
            pct     = float(meal.get("Amount Consumed", 100) or 100)
        except (ValueError, TypeError):
            continue

        # 섭취 비율 적용
        ratio = max(0.1, min(pct / 100.0, 1.0))
        carbs   *= ratio
        protein *= ratio
        fat     *= ratio
        fiber   *= ratio
        cal     *= ratio

        meal_type = str(meal.get("Meal Type", "")).strip().lower()
        if meal_type not in ("breakfast", "lunch", "dinner"):
            meal_type = "lunch"

        net_carbs = max(0.0, carbs - fiber)

        records.append({
            "pre_bg":      round(pre_bg, 1),
            "bg_rise":     round(bg_rise, 1),
            "carbs_g":     round(carbs,   1),
            "protein_g":   round(protein, 1),
            "fat_g":       round(fat,     1),
            "fiber_g":     round(fiber,   1),
            "net_carbs_g": round(net_carbs, 1),
            "calories":    round(cal,     1),
            "meal_type":   meal_type,
        })

    return records


def load_cgmacros(cgmacros_zip_path: str) -> pd.DataFrame:
    """
    CGMacros zip에서 모든 참여자의 식사별 BG rise 데이터를 로드합니다.

    Parameters
    ----------
    cgmacros_zip_path : CGMacros_dateshifted365.zip 경로

    Returns
    -------
    DataFrame with columns:
      pre_bg, bg_rise, carbs_g, protein_g, fat_g,
      fiber_g, net_carbs_g, calories, meal_type
    """
    all_records = []
    participant_pattern = re.compile(r"CGMacros/CGMacros-\d+/CGMacros-\d+\.csv$", re.IGNORECASE)

    with zipfile.ZipFile(cgmacros_zip_path) as z:
        csv_files = [n for n in z.namelist() if participant_pattern.search(n)]
        csv_files.sort()

        for fname in csv_files:
            try:
                with z.open(fname) as f:
                    df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
                records = _extract_meals_from_df(df)
                all_records.extend(records)
            except Exception:
                continue

    result = pd.DataFrame(all_records)
    return result
