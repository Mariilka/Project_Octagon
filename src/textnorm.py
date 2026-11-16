"""Нормализация слов.

По умолчанию — встроенный стеммер Портера для русского языка (алгоритм Snowball),
чтобы код работал без дополнительных пакетов. Если установлен pymorphy3
(pip install pymorphy3), можно включить настоящую лемматизацию: normalizer("lemma").
"""
import re

_V = "аеиоуыэюя"
_PERF_GER = re.compile(r"((?<=[ая])(в|вши|вшись)|(ив|ивши|ившись|ыв|ывши|ывшись))$")
_REFL = re.compile(r"(с[яь])$")
_ADJ = re.compile(r"(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$")
_PART = re.compile(r"((?<=[ая])(ем|нн|вш|ющ|щ)|(ивш|ывш|ующ))$")
_VERB = re.compile(
    r"((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)|"
    r"(ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю))$")
_NOUN = re.compile(r"(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$")
_DERIV = re.compile(r"(ост|ость)$")
_SUPER = re.compile(r"(ейш|ейше)$")
_RV = re.compile(r"^(.*?[аеиоуыэюя])(.*)$")
_R2 = re.compile(r"^.*?[аеиоуыэюя][^аеиоуыэюя].*?[аеиоуыэюя][^аеиоуыэюя](.*)$")


def stem(word: str) -> str:
    word = word.lower().replace("ё", "е")
    m = _RV.match(word)
    if not m:
        return word
    pre, rv = m.groups()
    new = _PERF_GER.sub("", rv, 1)
    if new == rv:
        rv = _REFL.sub("", rv, 1)
        new = _ADJ.sub("", rv, 1)
        if new != rv:
            rv = _PART.sub("", new, 1)
        else:
            new = _VERB.sub("", rv, 1)
            rv = new if new != rv else _NOUN.sub("", rv, 1)
    else:
        rv = new
    rv = re.sub(r"и$", "", rv, 1)
    r2 = _R2.match(pre + rv)
    if r2 and _DERIV.search(r2.group(1)):
        rv = _DERIV.sub("", rv, 1)
    new = re.sub(r"ь$", "", rv, 1)
    if new == rv:
        rv = _SUPER.sub("", rv, 1)
        rv = re.sub(r"нн$", "н", rv, 1)
    else:
        rv = new
    return pre + rv


def normalizer(kind: str = "stem"):
    """Возвращает функцию слово -> нормальная форма."""
    if kind == "lemma":
        import pymorphy3  # необязательная зависимость
        morph = pymorphy3.MorphAnalyzer()
        cache = {}

        def lemma(w):
            if w not in cache:
                cache[w] = morph.parse(w)[0].normal_form
            return cache[w]
        return lemma
    return stem


TOKEN = re.compile(r"[а-яё]{3,}")


def tokens(text: str):
    return TOKEN.findall(text.lower())
