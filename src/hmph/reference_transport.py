"""Metered, destination-restricted transport for isolated third-party research code.

Native configurations use a harmless placeholder. Only this HTTPS transport adds
the local credential, and it never persists headers, prompts or provider errors.
"""

import asyncio
import json
import threading
import time

import httpx
from pydantic import SecretStr


class ReferenceCallError(Exception):
    pass


class ReferenceTransport(httpx.BaseTransport):
    def __init__(
        self,
        key: SecretStr,
        *,
        model="deepseek-flash",
        max_calls=12,
        max_tokens=1024,
        delegate=None,
    ):
        self._key = key
        self.model, self.max_calls, self.max_tokens = model, max_calls, max_tokens
        self.delegate = delegate or httpx.HTTPTransport(retries=0)
        self.records = []
        self.budget_exhausted = False
        self._lock = threading.Lock()

    def handle_request(self, request):
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
        try:
            response = self.delegate.handle_request(outbound)
            response.read()
            record["http_status"] = response.status_code
            if response.status_code != 200:
                response.close()
                raise ReferenceCallError("reference_provider_error")
            if len(response.content) > 65536:
                response.close()
                raise ReferenceCallError("reference_response_limit")
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
            record["status"] = "completed"
            return response
        except Exception as error:
            code = (
                str(error)
                if isinstance(error, ReferenceCallError)
                else "reference_network_or_protocol"
            )
            record.update(status="failed", error_code=code)
            raise ReferenceCallError(code) from None
        finally:
            record["seconds"] = round(time.monotonic() - started, 3)

    def close(self):
        self.delegate.close()


class AsyncReferenceTransport(httpx.AsyncBaseTransport):
    """Async SDK adapter sharing the synchronous transport's limits and call ledger."""

    def __init__(self, key, **kwargs):
        self._transport = ReferenceTransport(key, **kwargs)
        self.records = self._transport.records

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
