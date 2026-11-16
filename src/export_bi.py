"""Шаг 4. Выгрузка витрины для BI-системы (Yandex DataLens и др.).

Запуск:  python src/export_bi.py
Результат: outputs/bi/reviews_bi.csv — одна строка = один отзыв, понятные русские названия полей.
Имена авторов и тексты отзывов в витрину не попадают.
"""
from pathlib import Path

import pandas as pd

import lexicon as lx
import sys

sys.stdout.reconfigure(errors="replace")  # консоль Windows (cp1251) не падает на символах вне кодировки

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "reviews_features.csv")
    out = pd.DataFrame({
        "ID отзыва": df["review_id"],
        "Источник": df["source"],
        "Компания": df["company"],
        "Год": df["year"],
        "Оценка": df["rating"],
        "Тональность": df["sentiment"],
        "Есть текст": df["has_text"].astype(int),
        "Длина, слов": df["n_words"],
        "Длина, символов": df["n_chars"],
        "Индекс эмоциональности": df["emo_index"].round(2),
        "Скорость доставки": df["delivery_speed"].where(df["has_text"], "Нет текста"),
        "Тип мебели": df["furniture_type"].where(df["has_text"], "Нет текста"),
        "Возврат или претензия": df["claim_event"].astype(int),
        "Цена в отзыве, руб": df["price_rub"],
        "Жалоба на сервис": (df["c_service"] & (df["rating"] <= 2)).astype(int),
        "Жалоба на качество": (df["c_quality"] & (df["rating"] <= 2)).astype(int),
    })
    for key, label in lx.ASPECT_LABELS.items():
        out[f"Тема: {label}"] = df[f"a_{key}"].astype(int)
    for i, name in enumerate(lx.CLAIM_CAUSES, 1):
        out[f"Причина претензии: {name}"] = (df[f"cause_{i}"] & df["claim_event"]).astype(int)
    dest = ROOT / "outputs" / "bi"
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / "reviews_bi.csv", index=False, encoding="utf-8-sig")
    print("Сохранено:", dest / "reviews_bi.csv", out.shape)

    # компактная витрина для файлового подключения Yandex DataLens (до 3–4 МБ, без запятых в значениях)
    causes = ["Брак, дефект, поломка", "Нарушение сроков", "Несоответствие заказу", "Некомплект", "Ошибки сборки", "Не подошёл"]
    claim = df["claim_event"] & (df["rating"] <= 3)
    dl = pd.DataFrame({
        "id": df["review_id"], "Источник": df["source"],
        "Компания": df["company"].str.replace(",", " ").str.replace('"', ""),
        "Год": df["year"].astype("Int64"), "Оценка": df["rating"], "Тональность": df["sentiment"],
        "Есть текст": df["has_text"].astype(int), "Слов": df["n_words"], "Эмоциональность": df["emo_index"].round(2),
        "Доставка": df["delivery_speed"].where(df["has_text"], "Нет текста"),
        "Тип мебели": df["furniture_type"].where(df["has_text"], "Нет текста"),
        "Претензия": claim.astype(int), "Цена": df["price_rub"].astype("Int64"),
        "Сервис": df["c_service"].astype(int), "Качество": df["c_quality"].astype(int),
        "Персонал": df["a_staff"].astype(int), "Доставка и сроки": df["a_delivery"].astype(int),
        "Сборка": df["a_assembly"].astype(int), "Возвраты": df["a_claims"].astype(int),
        "Несоответствие": df["a_mismatch"].astype(int), "Некомплект": df["a_incomplete"].astype(int),
        "Тема цена": df["a_price"].astype(int), "Ассортимент": df["a_assortment"].astype(int),
    })
    for i, name in enumerate(causes, 1):
        dl[f"Причина: {name}"] = (df[f"cause_{i}"] & claim).astype(int)
    dl.to_csv(dest / "reviews_datalens.csv", index=False, encoding="utf-8")
    print("Сохранено:", dest / "reviews_datalens.csv", dl.shape)


if __name__ == "__main__":
    main()
