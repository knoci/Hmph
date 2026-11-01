"""Source-preserving excerpt windows shared by retrieval adapters."""

import re


def context_window(content: str, start: int, end: int, *, max_chars=800):
    if not 0 <= start <= end <= len(content) or max_chars <= 0:
        raise ValueError("Invalid source window")
    if len(content) <= max_chars:
        return 0, len(content)
    center = (start + end) // 2
    left = max(0, min(center - max_chars // 2, len(content) - max_chars))
    # Prefer a sentence boundary, but never replace the exact source with a summary.
    boundaries = list(re.finditer(r"[。！？.!?\n]", content[max(0, left - 120) : left]))
    if boundaries:
        left = max(0, left - 120) + boundaries[-1].end()
    right = min(len(content), left + max_chars)
    if end > right:
        right = min(len(content), end)
        left = max(0, right - max_chars)
    trailing = list(
        re.finditer(r"[。！？.!?\n]", content[max(end, left + max_chars * 3 // 4) : right])
    )
    if trailing:
        right = max(end, left + max_chars * 3 // 4) + trailing[-1].end()
    return left, right


def best_window(content: str, terms, *, max_chars=450):
    if len(content) <= max_chars:
        return 0, len(content)
    occurrences = []
    candidates = []
    for term in dict.fromkeys(terms):
        if term:
            matches = [
                (m.start(), m.end(), term) for m in re.finditer(re.escape(term), content, re.I)
            ]
            occurrences.extend(matches)
            # A repeated generic prefix must not exhaust all candidate positions.
            candidates.extend(
                matches
                if len(matches) <= 16
                else [matches[i * (len(matches) - 1) // 15] for i in range(16)]
            )
    if not occurrences:
        return context_window(content, 0, 0, max_chars=max_chars)
    # Coverage across different terms wins over the earliest generic match.
    choices = []
    for start, end, _ in candidates[:512]:
        left, right = context_window(content, start, end, max_chars=max_chars)
        covered = {term for a, b, term in occurrences if left <= a and b <= right}
        choices.append((len(covered), -left, left, right))
    _, _, left, right = max(choices)
    return left, right


def budget_window(content, terms, *, max_chars=450, max_bytes=1000):
    if max_bytes <= 0:
        raise ValueError("Invalid source byte budget")
    while True:
        start, end = best_window(content, terms, max_chars=max_chars)
        if len(content[start:end].encode("utf-8")) <= max_bytes:
            return start, end
        if max_chars == 1:
            raise ValueError("Source byte budget cannot hold one character")
        max_chars = max(1, max_chars * 4 // 5)
