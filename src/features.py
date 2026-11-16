"""Шаг 2. Разметка отзывов: тональность, темы, скорость доставки, тип мебели, цена.

Запуск:  python src/features.py
Результат: data/processed/reviews_features.csv
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

import lexicon as lx
import sys

sys.stdout.reconfigure(errors="replace")  # консоль Windows (cp1251) не падает на символах вне кодировки

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")
EMOJI = re.compile(lx.EMOJI)

# --- суммы в рублях: «45 000 руб», «45 тыс», «45т.р.», «200к», «1,5 млн» ---
NUM = r"(\d{1,3}(?:[  .]\d{3})+|\d+(?:[.,]\d+)?)"
PRICE_PATTERNS = [
    (re.compile(NUM + r"\s*(?:млн|миллион)"), 1_000_000),
    (re.compile(NUM + r"\s*(?:тыс|тысяч|т\.?\s?р\b|тр\b|к\b|k\b)"), 1_000),
    (re.compile(NUM + r"\s*(?:руб|р\.|р\b|₽)"), 1),
]
MIN_ITEM_PRICE = 5_000      # меньше — это обычно доставка/подъём/сборка, а не цена мебели
MAX_ITEM_PRICE = 3_000_000


def sentiment(rating):
    if rating <= 2:
        return "Негативный"
    if rating == 3:
        return "Нейтральный"
    return "Позитивный"


def extract_price(text):
    """Максимальная сумма, названная в отзыве (приближение стоимости покупки)."""
    found = []
    for pattern, mult in PRICE_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(1).replace(" ", "").replace(" ", "")
            if mult == 1 and re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
                raw = raw.replace(".", "")          # 45.000 руб -> 45000
            raw = raw.replace(",", ".")
            try:
                value = float(raw) * mult
            except ValueError:
                continue
            tail = text[m.end():m.end() + 18]
            if re.search(r"дешевле|дороже|больше|меньше|скидк", tail):
                continue                            # «на 20 тыс. дешевле» — разница, а не цена
            if MIN_ITEM_PRICE <= value <= MAX_ITEM_PRICE:
                found.append(value)
    return max(found) if found else np.nan


def main():
    df = pd.read_csv(PROC / "reviews_clean.csv")
    df["text"] = df["text"].fillna("")
    low = df["text"].str.lower().str.replace("ё", "е")

    df["sentiment"] = df["rating"].map(sentiment)

    # --- длина и подробность ---
    words = df["text"].map(lambda t: WORD.findall(t))
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = words.map(len)
    df["n_sentences"] = df["text"].map(lambda t: max(1, len(re.findall(r"[.!?…]+", t))) if t else 0)

    # --- темы ---
    for key, pattern in lx.ASPECTS.items():
        df[f"a_{key}"] = low.str.contains(pattern, regex=True)
    df["a_service"] = df[[f"a_{k}" for k in lx.SERVICE_ASPECTS]].any(axis=1)
    df["a_product"] = df[[f"a_{k}" for k in lx.PRODUCT_ASPECTS]].any(axis=1)
    df["n_aspects"] = df[[f"a_{k}" for k in lx.ASPECTS]].sum(axis=1)

    # --- жалобы ---
    # базовый вариант: тема отзыва, но фразы вида «к качеству претензий нет» не считаются жалобой на качество
    low_q = low.str.replace(lx.QUALITY_NEUTRALIZE, " ", regex=True)
    df["c_quality"] = low_q.str.contains(lx.QUALITY_COMPLAINT, regex=True)
    df["c_service"] = df["a_service"]
    # строгий вариант: только слова, которые сами означают проблему
    for key, pattern in lx.COMPLAINT_STRICT.items():
        src = low_q if key == "quality" else low
        df[f"s_{key}"] = src.str.contains(pattern, regex=True)
    df["s_service"] = df[["s_staff", "s_delivery", "s_assembly", "s_claims"]].any(axis=1)

    # --- скорость доставки (только если в отзыве есть тема доставки/сроков) ---
    low_d = low.str.replace(lx.DELIVERY_IGNORE, " ", regex=True)
    neg_fast = low_d.str.contains(lx.DELIVERY_NEG_FAST, regex=True)   # «не привезли в срок»
    neg_slow = low_d.str.contains(lx.DELIVERY_NEG_SLOW, regex=True)   # «без опозданий»
    low_d = low_d.str.replace(lx.DELIVERY_NEG_FAST, " ", regex=True).str.replace(lx.DELIVERY_NEG_SLOW, " ", regex=True)
    fast = (low_d.str.contains(lx.DELIVERY_FAST, regex=True) | neg_slow) & df["a_delivery"]
    slow = (low_d.str.contains(lx.DELIVERY_SLOW, regex=True) | neg_fast) & df["a_delivery"]
    df["delivery_speed"] = np.select(
        [fast & ~slow, slow & ~fast, fast & slow, df["a_delivery"]],
        ["Быстро / в срок", "Задержка", "Противоречиво", "Доставка без оценки скорости"],
        default="Доставка не упоминается",
    )

    # --- возвраты и претензии как событие и их причины ---
    low_c = low.str.replace(lx.CLAIM_NEUTRALIZE, " ", regex=True)
    df["claim_event"] = low_c.str.contains(lx.CLAIM_EVENT, regex=True)
    for i, (name, pattern) in enumerate(lx.CLAIM_CAUSES.items(), 1):
        src = low_q if i == 1 else (low_d if i == 2 else low)
        df[f"cause_{i}"] = src.str.contains(pattern, regex=True)

    # --- тип мебели ---
    soft = low.str.contains(lx.SOFT, regex=True)
    cab = low.str.contains(lx.CABINET, regex=True)
    df["furniture_type"] = np.select(
        [soft & ~cab, cab & ~soft, soft & cab],
        ["Мягкая", "Корпусная", "Обе"], default="Не указан")
    for name, pattern in lx.PRODUCTS.items():
        df[f"p_{name}"] = low.str.contains(pattern, regex=True)

    # --- эмоциональность ---
    nw = df["n_words"].replace(0, np.nan)
    df["n_excl"] = df["text"].str.count("!")
    df["n_emoji"] = df["text"].map(lambda t: len(EMOJI.findall(t)))
    df["n_caps"] = words.map(lambda ws: sum(1 for w in ws if len(w) >= 3 and w.isupper()))
    df["n_pos"] = low.str.count(lx.POSITIVE_WORDS)
    df["n_neg"] = low.str.count(lx.NEGATIVE_WORDS)
    df["n_strong"] = low.str.count(lx.STRONG_EMOTION)
    df["excl_per100"] = df["n_excl"] / nw * 100
    df["emo_words_per100"] = (df["n_pos"] + df["n_neg"]) / nw * 100
    # индекс эмоциональности: число эмоциональных маркеров на 100 слов
    df["emo_index"] = (df["n_excl"] + df["n_emoji"] + df["n_caps"] + df["n_strong"]) / nw * 100
    df["lex_polarity"] = (df["n_pos"] - df["n_neg"]) / (df["n_pos"] + df["n_neg"]).replace(0, np.nan)

    # --- цена ---
    df["price_rub"] = low.map(extract_price)

    df.to_csv(PROC / "reviews_features.csv", index=False)
    t = df[df["has_text"]]
    print("Отзывов с текстом:", len(t))
    print((t[[c for c in t.columns if c.startswith("a_")]].mean() * 100).round(1))
    print(t["delivery_speed"].value_counts())
    print(t["furniture_type"].value_counts())
    print("С ценой:", t["price_rub"].notna().sum())


if __name__ == "__main__":
    main()
