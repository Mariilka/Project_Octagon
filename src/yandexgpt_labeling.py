"""Необязательный шаг: разметка выборки отзывов с помощью YandexGPT.

Нужен, чтобы независимо проверить словарную разметку (жалоба на сервис / на качество).
В основном расчёте НЕ используется: все цифры отчёта получены словарным методом.

Перед запуском задайте переменные окружения:
    YC_API_KEY   — API-ключ сервисного аккаунта Yandex Cloud (роль ai.languageModels.user)
    YC_FOLDER_ID — идентификатор каталога

Запуск:  python src/yandexgpt_labeling.py --n 200
Результат: validation/yandexgpt_labels.csv и сравнение со словарём в консоли.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import sys

sys.stdout.reconfigure(errors="replace")  # консоль Windows (cp1251) не падает на символах вне кодировки

ROOT = Path(__file__).resolve().parents[1]
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
SYSTEM = (
    "Ты размечаешь отзывы о мебельных магазинах. Ответь строго JSON-объектом вида "
    '{"service": 0 или 1, "quality": 0 или 1}. '
    "service = 1, если автор жалуется на сервис: персонал, обслуживание, доставку, сроки, сборку, "
    "работу с возвратами и претензиями. quality = 1, если автор жалуется на качество самой мебели: "
    "брак, дефекты, поломки, плохие материалы. Никакого текста, кроме JSON."
)


def ask(text, api_key, folder_id):
    body = {
        "modelUri": f"gpt://{folder_id}/yandexgpt-lite/latest",
        "completionOptions": {"stream": False, "temperature": 0, "maxTokens": 50},
        "messages": [{"role": "system", "text": SYSTEM}, {"role": "user", "text": text[:3000]}],
    }
    r = requests.post(URL, headers={"Authorization": f"Api-Key {api_key}"}, json=body, timeout=60)
    r.raise_for_status()
    answer = r.json()["result"]["alternatives"][0]["message"]["text"]
    answer = answer.strip().strip("`").replace("json", "", 1).strip()
    return json.loads(answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="сколько негативных отзывов разметить")
    args = parser.parse_args()
    api_key, folder_id = os.environ["YC_API_KEY"], os.environ["YC_FOLDER_ID"]

    df = pd.read_csv(ROOT / "data" / "processed" / "reviews_features.csv")
    sample = df[(df["rating"] <= 2) & df["has_text"]].sample(args.n, random_state=1)
    rows = []
    for r in sample.itertuples():
        try:
            label = ask(r.text, api_key, folder_id)
        except Exception as e:  # сеть, квоты, неразобранный ответ
            print("пропуск", r.review_id, e)
            continue
        rows.append({"review_id": r.review_id, "gpt_service": int(label.get("service", 0)),
                     "gpt_quality": int(label.get("quality", 0)),
                     "rule_service": int(r.c_service), "rule_quality": int(r.c_quality)})
        time.sleep(0.2)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "validation" / "yandexgpt_labels.csv", index=False)
    for target in ("service", "quality"):
        agree = (res[f"gpt_{target}"] == res[f"rule_{target}"]).mean()
        print(f"{target}: GPT {res[f'gpt_{target}'].mean():.1%}, словарь {res[f'rule_{target}'].mean():.1%}, совпадение {agree:.1%}")


if __name__ == "__main__":
    main()
