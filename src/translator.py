"""
Korean ingredient name -> English translator.

Simple dictionary-based lookup. If a term is not in the dictionary,
the original input is returned as-is so English inputs still work.
"""

KO_TO_EN: dict[str, str] = {
    # 채소
    "감자": "potato",
    "고구마": "sweet potato",
    "양파": "onion",
    "파": "green onion",
    "대파": "green onion",
    "쪽파": "scallion",
    "당근": "carrot",
    "오이": "cucumber",
    "애호박": "zucchini",
    "호박": "pumpkin",
    "가지": "eggplant",
    "토마토": "tomato",
    "방울토마토": "cherry tomato",
    "브로콜리": "broccoli",
    "콜리플라워": "cauliflower",
    "시금치": "spinach",
    "상추": "lettuce",
    "양배추": "cabbage",
    "배추": "napa cabbage",
    "무": "radish",
    "마늘": "garlic",
    "생강": "ginger",
    "고추": "chili pepper",
    "피망": "bell pepper",
    "파프리카": "bell pepper",
    "버섯": "mushroom",
    "표고버섯": "shiitake mushroom",
    "느타리버섯": "oyster mushroom",
    "팽이버섯": "enoki mushroom",
    "셀러리": "celery",
    "아스파라거스": "asparagus",
    "옥수수": "corn",
    "콩나물": "bean sprouts",
    "숙주": "mung bean sprouts",
    "두부": "tofu",
    # 육류
    "닭고기": "chicken",
    "닭": "chicken",
    "닭가슴살": "chicken breast",
    "닭다리": "chicken leg",
    "돼지고기": "pork",
    "삼겹살": "pork belly",
    "소고기": "beef",
    "쇠고기": "beef",
    "다진고기": "ground beef",
    "베이컨": "bacon",
    "햄": "ham",
    "소시지": "sausage",
    "양고기": "lamb",
    # 해산물
    "연어": "salmon",
    "참치": "tuna",
    "새우": "shrimp",
    "오징어": "squid",
    "문어": "octopus",
    "조개": "clam",
    "꽃게": "crab",
    "고등어": "mackerel",
    "대구": "cod",
    "멸치": "anchovy",
    # 유제품 / 달걀
    "달걀": "egg",
    "계란": "egg",
    "우유": "milk",
    "버터": "butter",
    "치즈": "cheese",
    "크림": "cream",
    "요거트": "yogurt",
    # 곡물 / 탄수화물
    "쌀": "rice",
    "현미": "brown rice",
    "밀가루": "flour",
    "빵": "bread",
    "파스타": "pasta",
    "국수": "noodles",
    "면": "noodles",
    # 콩류
    "두부": "tofu",
    "콩": "beans",
    "렌틸콩": "lentils",
    "병아리콩": "chickpeas",
    # 견과류 / 오일
    "올리브오일": "olive oil",
    "참기름": "sesame oil",
    "식용유": "vegetable oil",
    "아몬드": "almonds",
    "호두": "walnuts",
    "땅콩": "peanuts",
    # 소스 / 양념
    "간장": "soy sauce",
    "된장": "miso",
    "고추장": "gochujang",
    "식초": "vinegar",
    "설탕": "sugar",
    "소금": "salt",
    "후추": "black pepper",
    "케첩": "ketchup",
    "마요네즈": "mayonnaise",
    "머스타드": "mustard",
    # 과일
    "사과": "apple",
    "배": "pear",
    "바나나": "banana",
    "딸기": "strawberry",
    "포도": "grapes",
    "레몬": "lemon",
    "오렌지": "orange",
    "아보카도": "avocado",
}


def translate_ingredient(name: str) -> str:
    """
    Translate a Korean ingredient name to English.
    Returns original string if not found (supports English input directly).
    """
    name = name.strip()
    return KO_TO_EN.get(name, name)


def translate_ingredients(names: list[str]) -> list[str]:
    """Translate a list of ingredient names (Korean or English) to English."""
    return [translate_ingredient(n) for n in names]
