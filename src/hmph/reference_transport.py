"""Metered, destination-restricted transport for isolated third-party research code.

Native configurations use a harmless placeholder. Only this HTTPS transport adds
the local credential. Default execution never persists headers or prompts.
An explicit public-research observer may retain a failed request/choice, without headers.
"""

import asyncio
import json
import threading
import time

import httpx
from pydantic import SecretStr


class ReferenceCallError(Exception):
    pass


def failure_code(error):
    """Never persist provider exception text, URLs or headers."""
    if isinstance(error, ReferenceCallError):
        from hmph.eval_budget import ERROR_CODES

        return str(error) if str(error) in ERROR_CODES else "reference_network_or_protocol"
    if isinstance(error, httpx.TimeoutException):
        return "reference_timeout"
    if isinstance(error, httpx.ProtocolError | ValueError | KeyError | TypeError | AttributeError):
        return "reference_protocol"
    if isinstance(error, httpx.TransportError):
        return "reference_network"
    return "reference_network_or_protocol"


def http_failure_code(status):
    return "reference_service_unavailable" if status == 503 else "reference_provider_error"


class ReferenceTransport(httpx.BaseTransport):
    def __init__(
        self,
        key: SecretStr,
        *,
        model="deepseek-flash",
        max_calls=12,
        max_tokens=1024,
        delegate=None,
        response_validator=None,
        budget=None,
        stop_file=None,
        failure_observer=None,
    ):
        self._key = key
        self.model, self.max_calls, self.max_tokens = model, max_calls, max_tokens
        self.delegate = delegate or httpx.HTTPTransport(retries=0)
        self.response_validator = response_validator
        self.budget = budget
        self.stop_file = stop_file
        self.failure_observer = failure_observer
        self.records = []
        self.budget_exhausted = False
        self._lock = threading.Lock()

    def handle_request(self, request):
        if self.stop_file is not None and self.stop_file.exists():
            raise ReferenceCallError("reference_cancelled")
        if (
            request.url.scheme != "https"
            or request.url.host != "api.deepseek.com"
            or request.url.port not in (None, 443)
            or request.url.path not in ("/chat/completions", "/v1/chat/completions")
            or request.method != "POST"
        ):
            raise ReferenceCallError("reference_destination_denied")
        try:
            body = json.loads(request.content)
            if body.get("model") != self.model or body.get("stream"):
                raise ReferenceCallError("reference_request_denied")
            body["thinking"] = {"type": "disabled"}
            body["max_tokens"] = min(int(body.get("max_tokens", self.max_tokens)), self.max_tokens)
            raw = json.dumps(body, ensure_ascii=False).encode()
            if len(raw) > 64000:
                raise ReferenceCallError("reference_input_limit")
        except (ValueError, TypeError, AttributeError):
            raise ReferenceCallError("reference_request_denied") from None
        with self._lock:
            if len(self.records) >= self.max_calls:
                self.budget_exhausted = True
                raise ReferenceCallError("reference_call_budget")
            reservation = None
            if self.budget is not None:
                try:
                    reservation = self.budget.acquire(len(raw))
                except Exception:
                    self.budget_exhausted = True
                    raise ReferenceCallError("reference_call_budget") from None
            record = {
                "status": "started",
                "usage": None,
                "model": self.model,
                "max_tokens": body["max_tokens"],
                "input_bytes": len(raw),
            }
            self.records.append(record)
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"authorization", "content-length"}
        }
        headers["Authorization"] = "Bearer " + self._key.get_secret_value()
        outbound = httpx.Request(
            "POST", request.url, headers=headers, content=raw, extensions=request.extensions
        )
        started = time.monotonic()
        response = None
        record["failure_phase"] = "request"
        try:
            if self.stop_file is not None and self.stop_file.exists():
                raise ReferenceCallError("reference_cancelled")
            response = self.delegate.handle_request(outbound)
            record["http_status"] = response.status_code
            record["failure_phase"] = "response_body"
            response.read()
            if response.status_code != 200:
                response.close()
                raise ReferenceCallError(http_failure_code(response.status_code))
            if len(response.content) > 65536:
                response.close()
                raise ReferenceCallError("reference_response_limit")
            record["failure_phase"] = "response_protocol"
            payload = response.json()
            usage = payload.get("usage")
            if isinstance(usage, dict) and all(
                isinstance(usage.get(k), int) and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                record["usage"] = {
                    k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
            choices = payload.get("choices")
            if (
                not isinstance(choices, list)
                or not choices
                or any(not isinstance(c, dict) or c.get("finish_reason") != "stop" for c in choices)
            ):
                response.close()
                raise ReferenceCallError("reference_incomplete_output")
            if self.response_validator is not None:
                record["failure_phase"] = "response_validation"
                try:
                    self.response_validator(payload)
                except Exception:
                    response.close()
                    raise ReferenceCallError("reference_invalid_output") from None
            record["status"] = "completed"
            record.pop("failure_phase", None)
            return response
        except Exception as error:
            code = failure_code(error)
            record.update(status="failed", error_code=code)
            if self.failure_observer is not None:
                try:
                    choices = response.json().get("choices") if response is not None else None
                    diagnostic = {"reservation": reservation, "request": body, "choices": choices}
                    # Even an echoed credential must never reach a diagnostic callback.
                    if self._key.get_secret_value() in json.dumps(diagnostic, ensure_ascii=False):
                        record["failure_capture"] = "credential_redacted"
                    else:
                        self.failure_observer(diagnostic)
                        record["failure_capture"] = "captured"
                except Exception:
                    record["failure_capture"] = "unavailable"
            raise ReferenceCallError(code) from None
        finally:
            if response is not None and record["status"] != "completed":
                response.close()
            record["seconds"] = round(time.monotonic() - started, 3)
            if self.budget is not None:
                self.budget.finish(reservation, record)

    def close(self):
        self.delegate.close()


class AsyncReferenceTransport(httpx.AsyncBaseTransport):
    """Async SDK adapter sharing the synchronous transport's limits and call ledger."""

    def __init__(self, key, **kwargs):
        self._transport = ReferenceTransport(key, **kwargs)
        self.records = self._transport.records

    @property
    def budget_exhausted(self):
        return self._transport.budget_exhausted

    async def handle_async_request(self, request):
        await request.aread()
        response = await asyncio.to_thread(self._transport.handle_request, request)
        try:
            return httpx.Response(
                response.status_code,
                # read() already decoded the provider body. Do not decode it twice.
                headers={
                    name: value
                    for name, value in response.headers.items()
                    if name.lower()
                    not in {"content-encoding", "content-length", "transfer-encoding"}
                },
                content=response.content,
                extensions=response.extensions,
            )
        finally:
            response.close()

    async def aclose(self):
        await asyncio.to_thread(self._transport.close)
