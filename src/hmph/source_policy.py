"""Conservative source exclusions shared by persistent memory writers."""

import re

SECRET = re.compile(r"\bsk-[a-zA-Z0-9_-]{16,}|-----BEGIN .*PRIVATE KEY-----", re.I)
NO_RETAIN = re.compile(r"不要记|别记|不用记|别存|不要保存|不许记|do not (?:remember|store)", re.I)


def retainable(content: str) -> bool:
    return not SECRET.search(content) and not NO_RETAIN.search(content)
