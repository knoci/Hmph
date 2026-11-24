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
    r"\b(?:hiking|hike|bike ride|trail|outdoors|outdoor|cycling)\b|徒步|骑行|户外", re.I
)


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
    if relation is not None:
        mention = re.compile(RELATIONS[relation] + RELATION_ALIASES.get(relation, ""), re.I)
        cue += f" I went somewhere or did an activity with my {relation}."
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
        candidates.append((i, bias))
    return tuple(candidates), cue
