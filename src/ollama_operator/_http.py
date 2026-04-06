"""Low-level HTTP helpers for communicating with the Ollama API.

This module provides thin wrappers around ``httpx`` for both synchronous and
asynchronous usage, including support for NDJSON streaming responses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from .exceptions import ConnectionError, ResponseError, TimeoutError

logger = logging.getLogger("ollama_operator")

_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)


def _build_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _handle_error(exc: Exception) -> None:
    """Translate httpx exceptions into ollama_operator exceptions."""
    if isinstance(exc, httpx.ConnectError):
        raise ConnectionError(
            f"Cannot connect to Ollama server: {exc}"
        ) from exc
    if isinstance(exc, httpx.TimeoutException):
        raise TimeoutError(f"Request timed out: {exc}") from exc
    raise exc


def _check_response(response: httpx.Response) -> None:
    """Raise :class:`ResponseError` for non-2xx status codes."""
    if response.is_success:
        return
    try:
        body = response.json()
        message = body.get("error", response.text)
    except Exception:
        message = response.text
    raise ResponseError(response.status_code, message)


# ---------------------------------------------------------------------------
# Synchronous helpers
# ---------------------------------------------------------------------------

class SyncHTTP:
    """Synchronous HTTP transport backed by ``httpx.Client``."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = httpx.Client(timeout=timeout or _DEFAULT_TIMEOUT)

    # -- lifecycle --

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SyncHTTP:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- request methods --

    def get(self, path: str) -> dict[str, Any]:
        url = _build_url(self.base_url, path)
        try:
            resp = self._client.get(url)
        except Exception as exc:
            _handle_error(exc)
            raise  # unreachable – keeps mypy happy
        _check_response(resp)
        return resp.json()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = _build_url(self.base_url, path)
        try:
            resp = self._client.post(url, json=payload)
        except Exception as exc:
            _handle_error(exc)
            raise
        _check_response(resp)
        return resp.json()

    def post_stream(
        self, path: str, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """POST with NDJSON streaming response."""
        url = _build_url(self.base_url, path)
        try:
            with self._client.stream("POST", url, json=payload) as resp:
                _check_response(resp)
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON line: %s", line)
        except (ResponseError, ConnectionError, TimeoutError):
            raise
        except Exception as exc:
            _handle_error(exc)
            raise

    def delete(self, path: str, payload: dict[str, Any]) -> None:
        url = _build_url(self.base_url, path)
        try:
            resp = self._client.request("DELETE", url, json=payload)
        except Exception as exc:
            _handle_error(exc)
            raise
        _check_response(resp)

    def head(self, path: str) -> int:
        """Send a HEAD request and return the HTTP status code."""
        url = _build_url(self.base_url, path)
        try:
            resp = self._client.head(url)
        except Exception as exc:
            _handle_error(exc)
            raise
        return resp.status_code


# ---------------------------------------------------------------------------
# Asynchronous helpers
# ---------------------------------------------------------------------------

class AsyncHTTP:
    """Asynchronous HTTP transport backed by ``httpx.AsyncClient``."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout or _DEFAULT_TIMEOUT)

    # -- lifecycle --

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncHTTP:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # -- request methods --

    async def get(self, path: str) -> dict[str, Any]:
        url = _build_url(self.base_url, path)
        try:
            resp = await self._client.get(url)
        except Exception as exc:
            _handle_error(exc)
            raise
        _check_response(resp)
        return resp.json()

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = _build_url(self.base_url, path)
        try:
            resp = await self._client.post(url, json=payload)
        except Exception as exc:
            _handle_error(exc)
            raise
        _check_response(resp)
        return resp.json()

    async def post_stream(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """POST with NDJSON streaming response."""
        url = _build_url(self.base_url, path)
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                _check_response(resp)
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed JSON line: %s", line)
        except (ResponseError, ConnectionError, TimeoutError):
            raise
        except Exception as exc:
            _handle_error(exc)
            raise

    async def delete(self, path: str, payload: dict[str, Any]) -> None:
        url = _build_url(self.base_url, path)
        try:
            resp = await self._client.request("DELETE", url, json=payload)
        except Exception as exc:
            _handle_error(exc)
            raise
        _check_response(resp)

    async def head(self, path: str) -> int:
        """Send a HEAD request and return the HTTP status code."""
        url = _build_url(self.base_url, path)
        try:
            resp = await self._client.head(url)
        except Exception as exc:
            _handle_error(exc)
            raise
        return resp.status_code
