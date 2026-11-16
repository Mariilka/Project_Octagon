"""Шаг 3. Первичный анализ и проверка пяти гипотез.

Запуск:  python src/hypotheses.py
Результат: outputs/results.json, outputs/tables/*.csv, outputs/figures/*.png
"""
from collections import Counter, defaultdict
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import lexicon as lx
from textnorm import normalizer, tokens
import sys

sys.stdout.reconfigure(errors="replace")  # консоль Windows (cp1251) не падает на символах вне кодировки

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, BAD, GRAY = "#0ca30c", "#d03b3b", "#898781"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.dpi": 150,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

R = {}  # все числа, которые идут в отчёт и на монитор


# ---------------------------------------------------------------- статистика
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def share(mask):
    k, n = int(mask.sum()), int(len(mask))
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": round(k / n * 100, 1), "ci": [round(lo * 100, 1), round(hi * 100, 1)]}


def mcnemar(a, b):
    """Точный тест Мак-Нимара для двух признаков на одних и тех же отзывах."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "both": int((a & b).sum()),
            "neither": int((~a & ~b).sum()), "p": float(p)}


def odds_ratio(a, b, c, d):
    """OR и 95% ДИ для таблицы [[a, b], [c, d]] (поправка 0.5)."""
    a, b, c, d = (x + 0.5 for x in (a, b, c, d))
    or_ = a * d / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return round(or_, 1), [round(np.exp(np.log(or_) - 1.96 * se), 1), round(np.exp(np.log(or_) + 1.96 * se), 1)]


def ols(y, X, names):
    """МНК с классическими стандартными ошибками."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = resid @ resid / dof * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return {n: {"beta": round(float(b), 4), "se": round(float(s), 4), "p": float(pv)}
            for n, b, s, pv in zip(["const"] + names, beta, se, p)}


def fmt_p(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def barh(ax, labels, values, colors, fmt="{:.1f}%", xlim=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, xlim or max(values) * 1.18)
    for yi, v in zip(y, values):
        ax.text(v + (xlim or max(values)) * 0.012, yi, fmt.format(v).replace(".", ","), va="center", color=INK, fontsize=9)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- первичный анализ
def eda(df, t):
    R["data"] = {
        "n_reviews": int(len(df)), "n_text": int(len(t)), "n_companies": int(df["company"].nunique()),
        "years": [int(df["year"].min()), int(df["year"].max())],
        "mean_rating": round(float(df["rating"].mean()), 2),
        "rating_counts": {int(k): int(v) for k, v in df["rating"].value_counts().sort_index().items()},
        "sentiment_pct": (df["sentiment"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "by_source": df.groupby("source").agg(
            n=("rating", "size"), with_text=("has_text", "sum"), mean_rating=("rating", "mean"),
            neg_pct=("rating", lambda x: (x <= 2).mean() * 100),
            median_words=("n_words", lambda x: x[x > 0].median())).round(2).reset_index().to_dict("records"),
        "median_words": float(t["n_words"].median()),
        "median_words_neg": float(t.loc[t["rating"] <= 2, "n_words"].median()),
        "median_words_pos": float(t.loc[t["rating"] >= 4, "n_words"].median()),
        "google_truncated_pct": round(float(t.loc[t["source"] == "Google Карты", "truncated"].mean() * 100), 1),
    }
    by_year = df.groupby("year").agg(n=("rating", "size"), mean_rating=("rating", "mean"),
                                     neg_pct=("rating", lambda x: (x <= 2).mean() * 100)).round(2).reset_index()
    by_year.to_csv(TAB / "by_year.csv", index=False)
    R["data"]["by_year"] = by_year[by_year["year"] >= 2018].to_dict("records")

    # распределение оценок
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    counts = df["rating"].value_counts().sort_index()
    colors = [BAD, BAD, GRAY, GOOD, GOOD]
    ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.62)
    for x, v in zip(counts.index.astype(str), counts.values):
        ax.text(x, v + 120, f"{v:,}".replace(",", " "), ha="center", fontsize=9)
    ax.set_title("Распределение оценок"); ax.set_xlabel("Оценка, звёзд"); ax.set_ylabel("Отзывов")
    ax.grid(axis="x", visible=False)
    save(fig, "01_ratings.png")

    # темы в позитивных и негативных отзывах
    neg, pos = t[t["rating"] <= 2], t[t["rating"] >= 4]
    rows = []
    for key, label in lx.ASPECT_LABELS.items():
        rows.append({"Тема": label, "В негативных, %": round(neg[f"a_{key}"].mean() * 100, 1),
                     "В позитивных, %": round(pos[f"a_{key}"].mean() * 100, 1)})
    topics = pd.DataFrame(rows)
    topics["Во сколько раз чаще в негативе"] = (topics["В негативных, %"] / topics["В позитивных, %"]).round(1)
    topics.to_csv(TAB / "topics_by_sentiment.csv", index=False)
    R["topics"] = topics.to_dict("records")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(topics))[::-1]
    ax.barh(y + 0.19, topics["В негативных, %"], height=0.36, color=BAD, label="Негативные (1–2 балла)")
    ax.barh(y - 0.19, topics["В позитивных, %"], height=0.36, color=GOOD, label="Позитивные (4–5 баллов)")
    ax.set_yticks(y, topics["Тема"]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Доля отзывов, где тема упоминается, %"); ax.set_title("О чём пишут в негативных и позитивных отзывах")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "02_topics.png")

    # словарь жалоб: слова, характерные для негативных отзывов
    norm = normalizer("stem")
    doc_freq = {"neg": Counter(), "pos": Counter()}
    surface = defaultdict(Counter)
    for group, part in (("neg", neg), ("pos", pos)):
        for text in part["text"]:
            seen = set()
            for w in tokens(text):
                if w in lx.STOPWORDS:
                    continue
                s = norm(w)
                surface[s][w] += 1
                seen.add(s)
            doc_freq[group].update(seen)
    rows = []
    for s, k in doc_freq["neg"].items():
        if k < 40:
            continue
        p_neg = k / len(neg)
        p_pos = (doc_freq["pos"].get(s, 0) + 1) / (len(pos) + 1)
        rows.append({"Слово": surface[s].most_common(1)[0][0], "Основа": s, "Негативных отзывов": k,
                     "Доля в негативных, %": round(p_neg * 100, 1), "Доля в позитивных, %": round(p_pos * 100, 2),
                     "Во сколько раз чаще в негативе": round(p_neg / p_pos, 1)})
    words = pd.DataFrame(rows).sort_values(["Во сколько раз чаще в негативе", "Основа"], ascending=[False, True])  # порядок не зависит от запуска
    words.to_csv(TAB / "complaint_words.csv", index=False)
    top = words.head(25).sort_values(["Доля в негативных, %", "Основа"], ascending=[False, True])
    R["complaint_words"] = top.to_dict("records")
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    barh(ax, top["Слово"].tolist(), top["Доля в негативных, %"].tolist(), BAD)
    ax.set_title("Словарь жалоб: слова, типичные для негативных отзывов")
    ax.set_xlabel("Доля негативных отзывов со словом, % (отобраны по перевесу над позитивными)")
    save(fig, "03_complaint_words.png")

    # согласованность словаря тональности с оценкой (проверка лексикона)
    v = t[t["lex_polarity"].notna() & (t["rating"] != 3)]
    y_true = (v["rating"] >= 4).astype(int)
    u = stats.mannwhitneyu(v.loc[y_true == 1, "lex_polarity"], v.loc[y_true == 0, "lex_polarity"])
    R["lexicon_auc"] = round(float(u.statistic / (y_true.sum() * (1 - y_true).sum())), 3)


# ---------------------------------------------------------------- гипотеза 1
def h1(t):
    neg = t[t["rating"] <= 2]
    res = {"n_neg": int(len(neg))}
    for variant, s_col, q_col in (("base", "c_service", "c_quality"), ("strict", "s_service", "s_quality")):
        res[variant] = {"service": share(neg[s_col]), "quality": share(neg[q_col]),
                        "mcnemar": mcnemar(neg[s_col], neg[q_col])}
    res["staff_only"] = share(neg["a_staff"])
    res["staff_vs_quality"] = mcnemar(neg["a_staff"], neg["c_quality"])
    cats = [("Персонал и обслуживание", "a_staff", BLUE), ("Доставка и сроки", "a_delivery", BLUE),
            ("Возвраты и претензии", "a_claims", BLUE), ("Сборка и монтаж", "a_assembly", BLUE),
            ("Качество мебели (дефекты, материалы)", "c_quality", ORANGE),
            ("Несоответствие заказу/описанию", "a_mismatch", ORANGE), ("Некомплект", "a_incomplete", ORANGE),
            ("Цена", "a_price", GRAY), ("Ассортимент", "a_assortment", GRAY)]
    rows = [{"Причина": n, "Отзывов": int(neg[c].sum()), "Доля, %": round(neg[c].mean() * 100, 1),
             "Группа": {BLUE: "Сервис", ORANGE: "Товар", GRAY: "Прочее"}[col]} for n, c, col in cats]
    tab = pd.DataFrame(rows).sort_values("Доля, %", ascending=False)
    tab.to_csv(TAB / "h1_negative_reasons.csv", index=False)
    res["reasons"] = tab.to_dict("records")
    res["by_source"] = (neg.groupby("source").agg(n=("rating", "size"), service=("c_service", "mean"),
                                                  quality=("c_quality", "mean")) * [1, 100, 100]).round(1).reset_index().to_dict("records")
    res["validation"] = pd.read_csv(OUT / "validation_metrics.csv").to_dict("records")
    R["h1"] = res

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1.5]})
    ax = axes[0]
    x = np.arange(2); w = 0.36
    sv = [res["base"]["service"]["pct"], res["strict"]["service"]["pct"]]
    qv = [res["base"]["quality"]["pct"], res["strict"]["quality"]["pct"]]
    ax.bar(x - w / 2 - 0.01, sv, w, color=BLUE, label="Сервис"); ax.bar(x + w / 2 + 0.01, qv, w, color=ORANGE, label="Качество мебели")
    for xi, a, b in zip(x, sv, qv):
        ax.text(xi - w / 2, a + 1.5, f"{a:.1f}%".replace(".", ","), ha="center", fontsize=9)
        ax.text(xi + w / 2, b + 1.5, f"{b:.1f}%".replace(".", ","), ha="center", fontsize=9)
    ax.set_xticks(x, ["Базовая разметка\n(тема отзыва)", "Строгая разметка\n(явные жалобы)"], fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("Доля негативных отзывов, %"); ax.grid(axis="x", visible=False)
    ax.set_title("Сервис против качества"); ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    colors = [{"Сервис": BLUE, "Товар": ORANGE, "Прочее": GRAY}[g] for g in tab["Группа"]]
    barh(ax, tab["Причина"].tolist(), tab["Доля, %"].tolist(), colors, xlim=75)
    ax.set_title(f"Темы негативных отзывов (n = {len(neg)})"); ax.set_xlabel("Доля негативных отзывов, %")
    save(fig, "04_h1_service_vs_quality.png")


# ---------------------------------------------------------------- гипотеза 2
SMALL_DELAY = r"немного|небольш|чуть|на день|пару дней|не критич|незначит"   # задержка описана как небольшая


def h2(t):
    order = ["Быстро / в срок", "Доставка не упоминается", "Доставка без оценки скорости", "Противоречиво", "Задержка"]
    rows = []
    for g in order:
        part = t[t["delivery_speed"] == g]
        s = share(part["rating"] >= 4)
        rows.append({"Группа": g, "Отзывов": len(part), "Позитивных (4–5 баллов), %": s["pct"],
                     "ДИ 95%": f"{s['ci'][0]}–{s['ci'][1]}".replace(".", ","),
                     "Средняя оценка": round(part["rating"].mean(), 2),
                     "Доля оценок 5 баллов, %": round((part["rating"] == 5).mean() * 100, 1)})
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "h2_delivery_speed.csv", index=False)
    fast, slow = t[t["delivery_speed"] == "Быстро / в срок"], t[t["delivery_speed"] == "Задержка"]
    a, b = int((fast["rating"] >= 4).sum()), int((fast["rating"] < 4).sum())
    c, d = int((slow["rating"] >= 4).sum()), int((slow["rating"] < 4).sum())
    chi2, p, _, _ = stats.chi2_contingency([[a, b], [c, d]])
    or_, ci = odds_ratio(a, b, c, d)
    # отношение шансов Мантеля–Хензеля со стратификацией по источнику
    num = den = 0.0
    strata = []
    for src, part in t[t["delivery_speed"].isin(["Быстро / в срок", "Задержка"])].groupby("source"):
        f, s = part[part["delivery_speed"] == "Быстро / в срок"], part[part["delivery_speed"] == "Задержка"]
        a_, b_ = (f["rating"] >= 4).sum(), (f["rating"] < 4).sum()
        c_, d_ = (s["rating"] >= 4).sum(), (s["rating"] < 4).sum()
        n_ = a_ + b_ + c_ + d_
        num += a_ * d_ / n_; den += b_ * c_ / n_
        strata.append({"source": src, "fast_n": int(len(f)), "fast_pos_pct": round(a_ / max(len(f), 1) * 100, 1),
                       "slow_n": int(len(s)), "slow_pos_pct": round(c_ / max(len(s), 1) * 100, 1)})
    mw = stats.mannwhitneyu(fast["rating"], slow["rating"])
    base = t[~t["delivery_speed"].isin(["Быстро / в срок"])]
    R["h2"] = {
        "table": tab.to_dict("records"), "fast": share(fast["rating"] >= 4), "slow": share(slow["rating"] >= 4),
        "all_text_pos_pct": round(float((t["rating"] >= 4).mean() * 100), 1),
        "diff_pp": round(share(fast["rating"] >= 4)["pct"] - share(slow["rating"] >= 4)["pct"], 1),
        "chi2": round(float(chi2), 1), "p": float(p), "or": or_, "or_ci": ci,
        "or_mh": round(float(num / den), 1), "strata": strata,
        "mean_fast": round(float(fast["rating"].mean()), 2), "mean_slow": round(float(slow["rating"].mean()), 2),
        "mw_p": float(mw.pvalue),
        "slow_but_positive": int(c), "fast_but_negative": int((fast["rating"] <= 2).sum()),
        "small_delay_pos_pct": round(float(slow.loc[slow["rating"] >= 4, "text"].str.lower().str.contains(SMALL_DELAY).mean() * 100), 1),
        "small_delay_neg_pct": round(float(slow.loc[slow["rating"] <= 2, "text"].str.lower().str.contains(SMALL_DELAY).mean() * 100), 1),
        "delivery_mentioned_pct": round(float(t["a_delivery"].mean() * 100), 1),
        "speed_known_n": int(len(fast) + len(slow)),
        "neg_with_delay_pct": round(float((t.loc[t["rating"] <= 2, "delivery_speed"] == "Задержка").mean() * 100), 1),
    }
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    colors = [GOOD, GRAY, GRAY, GRAY, BAD]
    labels = [f"{g}\n(n = {n})" for g, n in zip(tab["Группа"], tab["Отзывов"])]
    barh(ax, labels, tab["Позитивных (4–5 баллов), %"].tolist(), colors, xlim=110)
    ax.set_xlabel("Доля положительных оценок (4–5 баллов), %"); ax.set_title("Скорость доставки и оценка")
    save(fig, "05_h2_delivery.png")


# ---------------------------------------------------------------- гипотеза 3
def h3(t):
    claims_all = t[t["claim_event"]]
    claims = claims_all[claims_all["rating"] <= 3]
    names = list(lx.CLAIM_CAUSES)
    cols = [f"cause_{i}" for i in range(1, len(names) + 1)]
    rows = []
    for n, c in zip(names, cols):
        s, s_all = share(claims[c]), share(claims_all[c])
        rows.append({"Причина": n, "Отзывов": s["k"], "Доля, %": s["pct"], "ДИ 95%": f"{s['ci'][0]}–{s['ci'][1]}".replace(".", ","),
                     "Доля среди всех оценок, %": s_all["pct"]})
    none = share(~claims[cols].any(axis=1))
    rows.append({"Причина": "Причина не определена", "Отзывов": none["k"], "Доля, %": none["pct"],
                 "ДИ 95%": f"{none['ci'][0]}–{none['ci'][1]}".replace(".", ","),
                 "Доля среди всех оценок, %": share(~claims_all[cols].any(axis=1))["pct"]})
    tab = pd.DataFrame(rows).sort_values("Доля, %", ascending=False)
    tab.to_csv(TAB / "h3_claim_causes.csv", index=False)
    R["h3"] = {
        "n_claims": int(len(claims)), "n_claims_all": int(len(claims_all)),
        "claims_pct_of_text": round(len(claims_all) / len(t) * 100, 1),
        "claims_pct_of_neg": round(float(t.loc[t["rating"] <= 2, "claim_event"].mean() * 100), 1),
        "table": tab.to_dict("records"),
        "defect_vs_mismatch": mcnemar(claims["cause_1"], claims["cause_3"]),
        "delay_vs_mismatch": mcnemar(claims["cause_2"], claims["cause_3"]),
        "mismatch_rank": int(tab.reset_index(drop=True).query("Причина == 'Несоответствие заказу/описанию'").index[0] + 1),
        "mismatch_in_all_neg_pct": round(float(t.loc[t["rating"] <= 2, "a_mismatch"].mean() * 100), 1),
    }
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colors = [ORANGE if n == "Несоответствие заказу/описанию" else (GRAY if n == "Причина не определена" else BLUE) for n in tab["Причина"]]
    barh(ax, tab["Причина"].tolist(), tab["Доля, %"].tolist(), colors, xlim=62)
    ax.set_title(f"Причины возвратов и претензий (n = {len(claims)})")
    ax.set_xlabel("Доля отзывов с возвратом/претензией, % (причин может быть несколько)")
    save(fig, "06_h3_claims.png")


# ---------------------------------------------------------------- гипотеза 4
def h4(t):
    g = t[t["furniture_type"].isin(["Мягкая", "Корпусная"])].copy()
    soft, cab = g[g["furniture_type"] == "Мягкая"], g[g["furniture_type"] == "Корпусная"]
    metrics = [("n_words", "Длина, слов"), ("n_chars", "Длина, символов"), ("emo_index", "Индекс эмоциональности (маркеров на 100 слов)"),
               ("excl_per100", "Восклицательных знаков на 100 слов"), ("emo_words_per100", "Оценочных слов на 100 слов")]
    rows, tests = [], {}
    for col, label in metrics:
        u = stats.mannwhitneyu(soft[col].dropna(), cab[col].dropna())
        ps = u.statistic / (soft[col].notna().sum() * cab[col].notna().sum())
        tests[col] = {"p": float(u.pvalue), "prob_soft_greater": round(float(ps), 3)}
        rows.append({"Показатель": label, "Мягкая: медиана": round(soft[col].median(), 2), "Корпусная: медиана": round(cab[col].median(), 2),
                     "Мягкая: среднее": round(soft[col].mean(), 2), "Корпусная: среднее": round(cab[col].mean(), 2),
                     "p (Манна–Уитни)": fmt_p(u.pvalue), "P(мягкая > корпусная)": round(float(ps), 3)})
    extra = [("Доля отзывов с «!» или эмодзи, %", lambda d: ((d["n_excl"] + d["n_emoji"]) > 0).mean() * 100),
             ("Доля крайних оценок (1 или 5 баллов), %", lambda d: d["rating"].isin([1, 5]).mean() * 100),
             ("Доля негативных (1–2 балла), %", lambda d: (d["rating"] <= 2).mean() * 100)]
    for label, f in extra:
        rows.append({"Показатель": label, "Мягкая: медиана": "", "Корпусная: медиана": "",
                     "Мягкая: среднее": round(f(soft), 1), "Корпусная: среднее": round(f(cab), 1), "p (Манна–Уитни)": "", "P(мягкая > корпусная)": ""})
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "h4_soft_vs_cabinet.csv", index=False)

    strat = g.groupby(["sentiment", "furniture_type"]).agg(n=("rating", "size"), median_words=("n_words", "median"),
                                                           median_emo=("emo_index", "median")).round(2).reset_index()
    strat.to_csv(TAB / "h4_by_sentiment.csv", index=False)
    by_src = g.groupby(["source", "furniture_type"]).agg(n=("rating", "size"), median_words=("n_words", "median"),
                                                         median_emo=("emo_index", "median")).round(2).reset_index()
    # регрессия: log(длина) и индекс эмоциональности с контролем источника и тональности
    X = np.column_stack([(g["furniture_type"] == "Мягкая").astype(float),
                         (g["source"] == "Google Карты").astype(float), (g["source"] == "Яндекс Карты").astype(float),
                         (g["sentiment"] == "Негативный").astype(float), (g["sentiment"] == "Нейтральный").astype(float)])
    names = ["soft", "google", "yandex", "negative", "neutral"]
    reg_len = ols(np.log(g["n_words"].clip(lower=1).values), X, names)
    reg_emo = ols(np.log1p(g["emo_index"].fillna(0).values), X, names)
    no_google = g[g["source"] != "Google Карты"]
    R["h4"] = {
        "n_soft": int(len(soft)), "n_cab": int(len(cab)), "n_both": int((t["furniture_type"] == "Обе").sum()),
        "n_unknown": int((t["furniture_type"] == "Не указан").sum()),
        "table": tab.to_dict("records"), "tests": tests,
        "by_sentiment": strat.to_dict("records"), "by_source": by_src.to_dict("records"),
        "reg_len_soft": reg_len["soft"], "reg_len_pct": round((np.exp(reg_len["soft"]["beta"]) - 1) * 100, 1),
        "reg_emo_soft": reg_emo["soft"], "reg_emo_pct": round((np.exp(reg_emo["soft"]["beta"]) - 1) * 100, 1),
        "no_google_median_words": no_google.groupby("furniture_type")["n_words"].median().to_dict(),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    sent = ["Позитивный", "Негативный"]
    x = np.arange(2); w = 0.36
    for ax, col, title, ylabel in ((axes[0], "median_words", "Длина отзыва", "Медиана, слов"),
                                   (axes[1], "median_emo", "Эмоциональность", "Медиана индекса (маркеров на 100 слов)")):
        sv = [strat[(strat["sentiment"] == s) & (strat["furniture_type"] == "Мягкая")][col].iloc[0] for s in sent]
        cv = [strat[(strat["sentiment"] == s) & (strat["furniture_type"] == "Корпусная")][col].iloc[0] for s in sent]
        ax.bar(x - w / 2 - 0.01, sv, w, color=BLUE, label="Мягкая мебель"); ax.bar(x + w / 2 + 0.01, cv, w, color=ORANGE, label="Корпусная мебель")
        for xi, a, b in zip(x, sv, cv):
            ax.text(xi - w / 2, a + max(sv + cv) * 0.02, f"{a:g}".replace(".", ","), ha="center", fontsize=9)
            ax.text(xi + w / 2, b + max(sv + cv) * 0.02, f"{b:g}".replace(".", ","), ha="center", fontsize=9)
        ax.set_xticks(x, [s + " отзыв" for s in sent]); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(sv + cv) * 1.18)
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "07_h4_soft_vs_cabinet.png")


# ---------------------------------------------------------------- гипотеза 5
def h5(t):
    p = t[t["price_rub"].notna()].copy()
    rho, pv = stats.spearmanr(p["price_rub"], p["n_words"])
    rho_c, pv_c = stats.spearmanr(p["price_rub"], p["n_aspects"])
    by_sent = {}
    for s, part in p.groupby("sentiment"):
        if len(part) >= 30:
            r_, p_ = stats.spearmanr(part["price_rub"], part["n_words"])
            by_sent[s] = {"n": int(len(part)), "rho": round(float(r_), 3), "p": float(p_)}
    p["tier"] = pd.qcut(p["price_rub"], 3, labels=["Нижняя треть", "Средняя треть", "Верхняя треть"])
    tiers = p.groupby("tier", observed=True).agg(n=("rating", "size"), price_from=("price_rub", "min"), price_to=("price_rub", "max"),
                                                 median_words=("n_words", "median"), median_aspects=("n_aspects", "median"),
                                                 neg_pct=("rating", lambda x: (x <= 2).mean() * 100)).round(1).reset_index()
    tiers.to_csv(TAB / "h5_price_tiers.csv", index=False)
    kw = stats.kruskal(*[g["n_words"].values for _, g in p.groupby("tier", observed=True)])
    # уровень товарных категорий: медианная цена категории и медианная длина отзыва о ней
    rows = []
    for name in lx.PRODUCTS:
        col = f"p_{name}"
        n_price = int(p[col].sum())
        rows.append({"Категория": name, "Отзывов": int(t[col].sum()), "Медиана длины, слов": float(t.loc[t[col], "n_words"].median()),
                     "Отзывов с ценой": n_price, "Медианная цена, руб.": float(p.loc[p[col], "price_rub"].median()) if n_price else np.nan})
    cat = pd.DataFrame(rows)
    cat.to_csv(TAB / "h5_categories.csv", index=False)
    cat_ok = cat[cat["Отзывов с ценой"] >= 15]
    rho_cat, p_cat = stats.spearmanr(cat_ok["Медианная цена, руб."], cat_ok["Медиана длины, слов"])
    R["h5"] = {
        "n_price": int(len(p)), "pct_price": round(len(p) / len(t) * 100, 1),
        "price_median": float(p["price_rub"].median()), "price_q1": float(p["price_rub"].quantile(.25)), "price_q3": float(p["price_rub"].quantile(.75)),
        "rho": round(float(rho), 3), "p": float(pv), "rho_aspects": round(float(rho_c), 3), "p_aspects": float(pv_c),
        "by_sentiment": by_sent, "tiers": tiers.to_dict("records"), "kruskal_p": float(kw.pvalue),
        "neg_pct_price": round(float((p["rating"] <= 2).mean() * 100), 1),
        "median_words_price": float(p["n_words"].median()), "median_words_noprice": float(t.loc[t["price_rub"].isna(), "n_words"].median()),
        "categories": cat.replace({np.nan: None}).to_dict("records"), "rho_cat": round(float(rho_cat), 2), "p_cat": float(p_cat), "n_cat": int(len(cat_ok)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    for s, col in (("Позитивный", GOOD), ("Нейтральный", GRAY), ("Негативный", BAD)):
        part = p[p["sentiment"] == s]
        ax.scatter(part["price_rub"], part["n_words"], s=22, color=col, alpha=0.75, edgecolor="white", linewidth=0.6, label=s)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([5e3, 2e4, 1e5, 5e5, 3e6], ["5 тыс.", "20 тыс.", "100 тыс.", "500 тыс.", "3 млн"])
    ax.set_yticks([10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1000"])
    ax.set_xlabel("Сумма, названная в отзыве, руб. (лог. шкала)"); ax.set_ylabel("Длина отзыва, слов (лог. шкала)")
    ax.set_title(f"Цена и длина отзыва (n = {len(p)}, ρ = {rho:.2f})".replace(".", ","))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    labels = [f"{r.tier}\n{r.price_from / 1000:.0f}–{r.price_to / 1000:.0f} тыс. руб." for r in tiers.itertuples()]
    ax.bar(labels, tiers["median_words"], color=BLUE, width=0.55)
    for x, v in zip(labels, tiers["median_words"]):
        ax.text(x, v + 2, f"{v:g}".replace(".", ","), ha="center", fontsize=9)
    ax.set_title("Медианная длина по ценовым третям"); ax.set_ylabel("Медиана, слов"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, tiers["median_words"].max() * 1.2)
    save(fig, "08_h5_price.png")


def main():
    for d in (FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC / "reviews_features.csv")
    df["text"] = df["text"].fillna("")
    t = df[df["has_text"]].copy()
    eda(df, t); h1(t); h2(t); h3(t); h4(t); h5(t)
    (OUT / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: R[k] for k in ("h1", "h2", "h3", "h4", "h5")}, ensure_ascii=False, indent=1, default=float)[:9000])


if __name__ == "__main__":
    main()
