"""Проверка словарной разметки на контрольной выборке негативных отзывов.

validation/control_labels.csv: 80 случайных отзывов с оценкой 1–2 (random_state=42),
для каждого по тексту отмечено (независимо от словаря), есть ли жалоба на сервис и на качество мебели
(разметка по первым 520 символам текста).
"""
from pathlib import Path

import pandas as pd

import lexicon as lx
import sys

sys.stdout.reconfigure(errors="replace")  # консоль Windows (cp1251) не падает на символах вне кодировки

ROOT = Path(__file__).resolve().parents[1]


def union(patterns):
    return "|".join(f"(?:{p})" for p in patterns)


def main():
    v = pd.read_csv(ROOT / "validation" / "control_labels.csv")
    low = v["text_520"].str.lower().str.replace("ё", "е")
    low_q = low.str.replace(lx.QUALITY_NEUTRALIZE, " ", regex=True)
    pred = {
        ("service", "базовый"): low.str.contains(union(lx.ASPECTS[k] for k in lx.SERVICE_ASPECTS)),
        ("quality", "базовый"): low_q.str.contains(lx.QUALITY_COMPLAINT),
        ("service", "строгий"): low.str.contains(union(lx.COMPLAINT_STRICT[k] for k in lx.SERVICE_ASPECTS)),
        ("quality", "строгий"): low_q.str.contains(lx.COMPLAINT_STRICT["quality"]),
    }
    rows = []
    for (target, variant), p in pred.items():
        m = v[f"manual_{target}"] == 1
        tp = (m & p).sum()
        rows.append({
            "Жалоба": "на сервис" if target == "service" else "на качество",
            "Вариант словаря": variant,
            "Доля по контрольной разметке, %": round(m.mean() * 100, 1),
            "Доля по словарю, %": round(p.mean() * 100, 1),
            "Точность": round(tp / p.sum(), 2),
            "Полнота": round(tp / m.sum(), 2),
            "Доля совпадений": round((m == p).mean(), 2),
        })
    res = pd.DataFrame(rows)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "validation_metrics.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
