"""Pair a withdrawal with an earlier permission about the same subject."""

import re

from hmph.lexical import tokenize

WITHDRAWAL = (
    r"(?<!未)(?<!不)撤(?:销|回|掉|下|一下)|(?<!不)收回|"
    r"(?<!未)(?<!不)(?<!没)(?<!没有)(?<!不会)作废|"
    r"\b(?:revoked?|withdrawn?|withdrew|rescinded?)\b"
)
_WITHDRAWAL = re.compile(WITHDRAWAL, re.I)
_PERMISSION = re.compile(r"(?<!不)(?:可以|允许|同意)|\b(?:may|can|allow(?:ed)?)\b", re.I)
_CLAUSE = re.compile(r"[^。！!？?；;\n]+[。！!？?；;]?")
_CONTRAST = re.compile(r"[，,]\s*(?:但是|但|不过|but\b)\s*", re.I)
_CONDITION = re.compile(
    r"除非|如果|假如|只要|仅当|直到|之前|\b(?:unless|if|only|until|before)\b", re.I
)
_UNCERTAIN = re.compile(r"如果|要是|假如|要不要|没有撤|\b(?:if|unless|not)\b", re.I)
_CURRENT = re.compile(
    r"现在|目前|还(?:能|可|应该|该|要)|仍|(?:能不能|可不可以|能否|是否可以)继续|"
    r"\b(?:now|still|currently|can I keep|can you keep)\b", re.I
)


def asks_current_state(query):
    return bool(_CURRENT.search(query)) and len(re.findall(r"[?？]", query)) <= 1


def _clauses(content):
    for clause in _CLAUSE.finditer(content):
        start = clause.start()
        for boundary in _CONTRAST.finditer(content, start, clause.end()):
            yield start, boundary.start()
            start = boundary.end()
        yield start, clause.end()


def _asserted_withdrawal(clause):
    return (
        not clause.endswith(("?", "？"))
        and _WITHDRAWAL.search(clause)
        and not _UNCERTAIN.search(clause)
    )


def _withdrawal_matches(content, terms):
    matched = set()
    for start, end in _clauses(content):
        clause = content[start:end]
        if _asserted_withdrawal(clause):
            shared = set(tokenize(clause)) & terms
            if len(shared) >= 2:
                matched.update(shared)
    return matched


def withdrawal_focus(sources, query):
    """Find direct withdrawals and adjacent conditions in their own conversation."""
    terms = set(tokenize(query))
    matched = set()
    conditions = {}
    positions = {source.source_order: source for source in sources}
    for source in sorted(sources, key=lambda item: item.source_order, reverse=True):
        shared = _withdrawal_matches(source.content, terms)
        if not shared:
            continue
        matched.update(shared)
        if not source.conversation_id:
            continue
        for offset in (1, -1):
            neighbor = positions.get(source.source_order + offset)
            if neighbor is None or neighbor.conversation_id != source.conversation_id:
                continue
            span = _neighbor_condition(neighbor.content, terms, shared, offset)
            if span is not None:
                conditions.setdefault(neighbor.message_id, span)
    return matched, conditions


def _neighbor_condition(content, query_terms, anchor_terms, offset):
    spans = list(_clauses(content))
    for index, (start, end) in enumerate(spans):
        clause = content[start:end]
        if not _CONDITION.search(clause):
            continue
        words = set(tokenize(clause))
        if index + 1 < len(spans):
            action_end = spans[index + 1][1]
            action = content[end:action_end]
            if len(set(tokenize(action)) & anchor_terms) >= 2:
                return start, action_end
        if words & (query_terms if offset == 1 else anchor_terms):
            return start, end
    return None


def withdrawal_window(content, terms, query):
    """Return an exact span of the matching action and any adjacent condition."""
    spans = list(_clauses(content))
    matches = []
    query_terms = set(tokenize(query))
    for index, (start, end) in enumerate(spans):
        text = content[start:end]
        words = set(tokenize(text))
        overlap = words & query_terms
        if len(overlap) >= 3 or (len(overlap) >= 2 and words & terms):
            matches.append(index)
    if not matches:
        return None
    left, right = matches[0], matches[-1]
    if left > 0 and _CONDITION.search(content[slice(*spans[left - 1])]):
        left -= 1
    if right + 1 < len(spans) and _CONDITION.search(content[slice(*spans[right + 1])]):
        right += 1
    return spans[left][0], spans[right][1]


def withdrawals(sources, query, *, anchors=None):
    """Return source index pairs in relevance order, without inventing a current fact."""
    query_terms = set(tokenize(query))
    # Match the action itself, not an unrelated sentence in the same message.
    permissions = []
    changes = []
    indices = range(len(sources)) if anchors is None else anchors
    for i in indices:
        for start, end in _clauses(sources[i].content):
            clause = sources[i].content[start:end]
            if not clause.endswith(("?", "？")) and _PERMISSION.search(clause):
                terms = set(tokenize(clause))
                if terms & query_terms:
                    permissions.append((i, terms))
    if not permissions:
        return ()
    for j, source in enumerate(sources):
        for start, end in _clauses(source.content):
            clause = source.content[start:end]
            if not _asserted_withdrawal(clause):
                continue
            changed = set(tokenize(clause))
            for i, prior in permissions:
                if i >= j:
                    continue
                shared = prior & changed & query_terms
                if len(shared) < 2:
                    continue
                changes.append((j, len(shared), i))
    # Begin with the latest update and keep at most one prior source for it.
    result = {}
    for j, _, i in sorted(changes, reverse=True):
        result.setdefault(j, i)
    return tuple((i, j) for j, i in result.items())[:2]
