"""Conservative source exclusions shared by persistent memory writers."""

import re

SECRET = re.compile(r"\bsk-[a-zA-Z0-9_-]{16,}|-----BEGIN .*PRIVATE KEY-----", re.I)
NO_RETAIN = re.compile(
    r"(?:不要|别|不用|不许)记(?!成|作|错|混)|别存|不要保存|do not (?:remember|store)", re.I
)


def retainable(content: str) -> bool:
    return not SECRET.search(content) and not NO_RETAIN.search(content)
