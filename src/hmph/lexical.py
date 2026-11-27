"""Unicode tokenization shared by full-text search and the local hash encoder."""

import math
import unicodedata
from collections import Counter
from itertools import groupby

TOKENIZER_VERSION = 2


STOP_WORDS = set(
    "a an the i me my we our you your he she it they them this that these those "
    "am is are was were be been being have has had do does did can could would should "
    "will may might and or but if so to of in on at for from with as by about "
    "what which who when where how many much ve m".split()
)


def _script_group(char: str) -> str:
    category = unicodedata.category(char)[0]
    if category not in {"L", "M", "N"}:
        return "separator"
    point = ord(char)
    if (
        0x1100 <= point <= 0x11FF
        or 0x3040 <= point <= 0x30FF
        or 0x31F0 <= point <= 0x31FF
        or 0x3400 <= point <= 0x4DBF
        or 0x4E00 <= point <= 0x9FFF
        or 0xAC00 <= point <= 0xD7FF
        or 0xF900 <= point <= 0xFAFF
        or 0x20000 <= point <= 0x323AF
    ):
        return "cjk"
    return "word"


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for script, chars in groupby(normalized, _script_group):
        if script == "separator":
            continue
        word = "".join(chars)
        if script == "cjk" and len(word) >= 2:
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
        elif word not in STOP_WORDS:
            tokens.append(word)
    return tokens


def bm25_order(documents, question):
    counts = [Counter(tokenize(document.content)) for document in documents]
    if not counts:
        return []
    sizes = [sum(c.values()) for c in counts]
    mean_size = sum(sizes) / len(sizes) or 1
    query = set(tokenize(question))
    frequency = Counter(token for count in counts for token in query if token in count)
    scores = []
    for index, count in enumerate(counts):
        score = 0
        for token in query:
            tf = count[token]
            if tf:
                idf = math.log(
                    1 + (len(counts) - frequency[token] + 0.5) / (frequency[token] + 0.5)
                )
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * sizes[index] / mean_size))
        if score:
            scores.append((score, index))
    return [index for _, index in sorted(scores, reverse=True)[:80]]
