"""Bounded, fixed-lifetime local browser sessions; restart invalidates all cookies."""

import hmac
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from hmph.contracts import DomainError

COOKIE = "hmph_session"


@dataclass(frozen=True)
class BrowserSession:
    csrf: str
    expires: float


class BrowserSessions:
    def __init__(self, ttl=28800):
        self.ttl = ttl
        self.sessions = {}
        self.attempts = {}

    @staticmethod
    def check_origin(request):
        origin = request.headers.get("origin", "")
        expected = str(request.base_url).rstrip("/")
        if (
            urlsplit(expected).hostname not in ("127.0.0.1", "localhost", "::1")
            or origin != expected
        ):
            raise DomainError("invalid_origin", 403)

    def login(self, request, token, expected_token):
        self.check_origin(request)
        now = time.monotonic()
        self.sessions = {k: v for k, v in self.sessions.items() if v.expires > now}
        self.attempts = {k: v for k, v in self.attempts.items() if v[0] > now - 60}
        host = request.client.host if request.client else "local"
        since, count = self.attempts.get(host, (now, 0))
        if count >= 10 or len(self.attempts) >= 128 and host not in self.attempts:
            raise DomainError("login_rate_limited", 429)
        self.attempts[host] = (since, count + 1)
        if not hmac.compare_digest(token.encode(), expected_token.encode()):
            raise DomainError("unauthorized", 401)
        if len(self.sessions) >= 64:
            raise DomainError("session_limit", 429)
        key = secrets.token_urlsafe(32)
        session = BrowserSession(secrets.token_urlsafe(32), now + self.ttl)
        self.sessions[key] = session
        return key, session

    def require(self, request, *, write=False):
        key = request.cookies.get(COOKIE, "")
        session = self.sessions.get(key)
        if session is None or session.expires <= time.monotonic():
            self.sessions.pop(key, None)
            raise DomainError("unauthorized", 401)
        if write:
            self.check_origin(request)
            supplied = request.headers.get("x-csrf-token", "")
            if not hmac.compare_digest(supplied.encode(), session.csrf.encode()):
                raise DomainError("invalid_csrf", 403)
        return session

    def logout(self, request):
        self.require(request, write=True)
        self.sessions.pop(request.cookies.get(COOKIE), None)
