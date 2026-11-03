"""Shared word/CJK ranking, extracted unchanged from the evaluation baseline."""

import math
import re
from collections import Counter

STOP_WORDS = set(
    "a an the i me my we our you your he she it they them this that these those "
    "am is are was were be been being have has had do does did can could would should "
    "will may might and or but if so to of in on at for from with as by about "
    "what which who when where how many much ve m".split()
)


def tokenize(text):
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower())
    return [
        part
        for word in words
        if word not in STOP_WORDS
        for part in (
            [word]
            if not re.fullmatch(r"[\u4e00-\u9fff]+", word) or len(word) < 2
            else [word[i : i + 2] for i in range(len(word) - 1)]
        )
    ]


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
