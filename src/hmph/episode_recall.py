"""Find participant-linked episodes in a frozen original-message snapshot."""

import re
from collections import Counter

LIST_QUESTION = re.compile(
    r"\b(?:what (?:kinds?|types?|sorts?) of|which (?:places|activities|classes|groups)|"
    r"what (?:places|activities|classes|groups))\b|哪些|做过什么|去过哪|参加过什么",
    re.I,
)
SPEAKER = re.compile(r"^([^\W\d_]\w{0,23}):\s")
RELATIONS = {
    "girlfriend": r"\bgirlfriend\b|女朋友|女友",
    "boyfriend": r"\bboyfriend\b|男朋友|男友",
    "wife": r"\bwife\b|妻子|老婆",
    "husband": r"\bhusband\b|丈夫|老公",
    "partner": r"\b(?:spouse|partner)\b|配偶|伴侣",
}
RELATION_ALIASES = {"girlfriend": r"|\b(?:gf|my girl)\b", "boyfriend": r"|\b(?:bf|my guy)\b"}
ENROLLMENT = re.compile(
    r"\b(?:class(?:es)?|courses?|groups?|workshops?|clubs?|training)\b|课程|班|小组|工作坊|培训",
    re.I,
)
PARTICIPATION = re.compile(
    r"\b(?:taking|took|joined|attended|enrolled|signed up|started|completed)\b|参加|加入|报名|上了",
    re.I,
)
OUTSIDE = re.compile(
    r"\b(?:hiking|hike|bike ride|trail|outdoors|outdoor|cycling|picnic|camping)\b"
    r"|徒步|骑行|户外|野餐|露营",
    re.I,
)
PLACE_QUERY = re.compile(r"\b(?:places|venues|locations|spots)\b|去过哪|哪些地方|地点", re.I)
ACTIVITY_QUERY = re.compile(r"\bactivities\b|做过什么|哪些活动", re.I)
VISIT = re.compile(
    r"\b(?:(?:went|walked|headed|drove|travel(?:led|ed)|cycled|rode)\s+(?:to|through)|"
    r"visited|checked out|tried out|stumbled (?:up)?on|volunteer(?:ed|ing)\s+at|"
    r"(?:stayed|dined|ate)\s+at|(?:found|discovered)\s+(?:a|the)\s+(?:new\s+)?"
    r"(?:place|spot|area|space))\b|去了|去过|参观了|逛了|发现了",
    re.I,
)
PAST_EVENT = re.compile(
    r"\b(?:went|visited|had|did|joined|took|started|tried|played|volunteer(?:ed|ing)|"
    r"hiked|saw|worked|made|spent)\b|已经|昨天|上周|上个月|参加了|做过|去过|去了",
    re.I,
)
FUTURE_EVENT = re.compile(
    r"\b(?:going (?:to|on)|planning to|plan to|will (?:go|visit)|would love to|"
    r"can.t wait (?:to|for))\b|准备去|打算去|计划去",
    re.I,
)
SENTENCE = re.compile(r"[^.!?。！？\n]+")


def episode_candidates(snapshot, query):
    """Return matching source positions and a first-person retrieval cue."""
    if not LIST_QUESTION.search(query):
        return (), None
    relation = next(
        (name for name, pattern in RELATIONS.items() if re.search(pattern, query, re.I)), None
    )
    courses = ENROLLMENT.search(query) is not None
    if relation is None and not courses:
        return (), None
    speakers = [SPEAKER.match(source.content) for source in snapshot]
    counts = Counter(match[1].casefold() for match in speakers if match)
    named = {
        name
        for name, count in counts.items()
        if count >= 2 and re.search(rf"\b{re.escape(name)}\b", query, re.I)
    }
    cue = query
    for name in sorted(named):
        cue = re.sub(rf"\b{re.escape(name)}\b(?:['’]s)?", "I", cue, flags=re.I)
    if named:
        cue = re.sub(r"\b(?:his|her|their)\b", "my", cue, flags=re.I)
    places = relation is not None and PLACE_QUERY.search(query) is not None
    if relation is not None:
        mention = re.compile(RELATIONS[relation] + RELATION_ALIASES.get(relation, ""), re.I)
        cue += (
            f" I visited a place or went to an event with my {relation}."
            if places
            else f" I went somewhere or did an activity with my {relation}."
        )
    else:
        mention = ENROLLMENT
        cue += " I took a class, joined a group, enrolled in a course, or attended a workshop."
    indoor = re.search(r"\bindoor\b|室内", query, re.I)
    candidates = []
    for i, (source, speaker) in enumerate(zip(snapshot, speakers, strict=True)):
        if named and speaker and speaker[1].casefold() not in named:
            continue
        body = source.content[speaker.end() :] if speaker else source.content
        if not mention.search(body):
            continue
        bias = 0.0
        if indoor and OUTSIDE.search(body):
            bias -= 0.12
        if courses and PARTICIPATION.search(body):
            bias += 0.10
        if places and any(
            VISIT.search(clause) and not FUTURE_EVENT.search(clause)
            for clause in SENTENCE.findall(body)
        ):
            bias += 0.18
        if (
            relation is not None
            and (places or ACTIVITY_QUERY.search(query))
            and FUTURE_EVENT.search(body)
            and not PAST_EVENT.search(body)
        ):
            bias -= 0.18
        candidates.append((i, bias))
    return tuple(candidates), cue


def event_terms(content):
    """Describe the event rather than the whole surrounding conversation."""
    from hmph.lexical import tokenize

    clauses = SENTENCE.findall(content)
    for pattern in (VISIT, PARTICIPATION, PAST_EVENT):
        terms = set(tokenize(" ".join(clause for clause in clauses if pattern.search(clause))))
        if terms:
            return terms
    return set(tokenize(content))
