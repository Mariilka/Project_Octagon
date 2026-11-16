"""Шаг 1. Загрузка, объединение и очистка отзывов из трёх источников.

Запуск:  python src/prepare.py
Результат: data/processed/reviews_clean.csv
"""
from pathlib import Path
import re

import pandas as pd
import sys

sys.stdout.reconfigure(errors="replace")  # консоль Windows (cp1251) не падает на символах вне кодировки

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SOURCES = {
    "2ГИС": "reviews_2gis.xlsx",
    "Google Карты": "reviews_google.xlsx",
    "Яндекс Карты": "reviews_yandex.xlsx",
}
# Год выгрузки данных: самые поздние даты в 2ГИС и Яндексе — март 2024.
SCRAPE_YEAR = 2024

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}


def parse_rating(value):
    """'5 звезд' / 5.0 / '4,8' -> число."""
    if pd.isna(value):
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return float(m.group().replace(",", ".")) if m else None


def parse_year(value):
    """Абсолютные даты ('6 марта 2024', '6 февраля') и относительные ('3 года назад')."""
    if pd.isna(value):
        return None
    s = str(value).lower()
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return int(m.group())
    m = re.search(r"(\d+)\s+(?:год|года|лет)\s+назад", s)
    if m:
        return SCRAPE_YEAR - int(m.group(1))
    if "год назад" in s:
        return SCRAPE_YEAR - 1
    if "назад" in s or any(mn in s for mn in MONTHS):
        return SCRAPE_YEAR  # «2 месяца назад», «6 февраля» без года
    return None


def clean_text(text):
    if pd.isna(text):
        return ""
    t = str(text).replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_name(name):
    n = re.sub(r"\s+", " ", str(name)).strip().strip(".").strip()
    return n


def load():
    frames = []
    for source, fname in SOURCES.items():
        df = pd.read_excel(RAW / fname)
        df["source"] = source
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    raw = load()
    log = {"Строк в исходных файлах": len(raw)}

    df = pd.DataFrame({
        "source": raw["source"],
        "company": raw["Наименование"].map(normalize_name),
        "address": raw["Адрес"],
        "author": raw["Автор"],
        "rating": raw["Оценка автора"].map(parse_rating),
        "year": raw["Дата"].map(parse_year),
        "text": raw["Текст"].map(clean_text),
        "likes": raw["Like"].fillna(0),
    })

    # 1) строки-заглушки без отзыва (карточка компании без автора и оценки)
    df = df[df["rating"].notna()].copy()
    log["После удаления строк без оценки"] = len(df)

    # 2) полные дубли и повторы одного текста одним автором
    #    (один магазин встречается в выгрузке несколько раз — по разным адресам/картам)
    with_text = df["text"].str.len() > 0
    dup = with_text & df.duplicated(["author", "text"], keep="first")
    df = df[~dup].copy()
    log["После удаления дублей (автор + текст)"] = len(df)

    df["has_text"] = df["text"].str.len() > 0
    # Google обрезает длинные отзывы многоточием — помечаем, чтобы учесть в гипотезах про длину
    df["truncated"] = (df["source"] == "Google Карты") & df["text"].str.endswith("…")
    df["rating"] = df["rating"].astype(int)
    df["year"] = df["year"].astype("Int64")
    df.insert(0, "review_id", range(1, len(df) + 1))

    log["Из них с текстом"] = int(df["has_text"].sum())
    log["Оценка без текста"] = int((~df["has_text"]).sum())

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "reviews_clean.csv", index=False)
    pd.Series(log).to_csv(OUT / "cleaning_log.csv", header=["value"])
    for k, v in log.items():
        print(f"{k}: {v}")
    print(df.groupby("source").agg(n=("rating", "size"), with_text=("has_text", "sum"),
                                   mean_rating=("rating", "mean")).round(2))


if __name__ == "__main__":
    main()
