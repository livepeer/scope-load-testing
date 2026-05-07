"""Async HTTP client for the Daydream SDK service.

Implements the same API that storyboard uses:
  /stream/start, /stream/{id}/status, /stream/{id}/frame,
  /stream/{id}/publish, /stream/{id}/control, /stream/{id}/stop
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SDKClient:
    """Async HTTP client for the Daydream SDK service."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SDKClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(connect=30.0, read=self._timeout, write=30.0, pool=30.0),
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("SDKClient not entered as async context manager")
        return self._client

    async def health(self) -> dict[str, Any]:
        resp = await self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def stream_start(self, params: dict[str, Any]) -> dict[str, Any]:
        """Start a stream. Returns {stream_id, publish_url, subscribe_url, ...}."""
        resp = await self.client.post(
            "/stream/start",
            json={"model_id": "scope", "params": params},
        )
        resp.raise_for_status()
        return resp.json()

    async def stream_status(self, stream_id: str) -> dict[str, Any] | None:
        """Get stream status. Returns None if stream not found (404)."""
        resp = await self.client.get(f"/stream/{stream_id}/status")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def stream_frame(self, stream_id: str) -> bytes | None:
        """Get the latest output frame. Returns None if no frame (204)."""
        resp = await self.client.get(f"/stream/{stream_id}/frame")
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return resp.content

    async def stream_publish(self, stream_id: str, jpeg_bytes: bytes, seq: int) -> bool:
        """Publish an input frame. Returns True on success."""
        resp = await self.client.post(
            f"/stream/{stream_id}/publish",
            params={"seq": seq},
            content=jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )
        return resp.is_success

    async def stream_control(self, stream_id: str, params: dict[str, Any]) -> bool:
        """Update stream parameters (prompt, noise, etc). Returns True on success."""
        resp = await self.client.post(f"/stream/{stream_id}/control", json=params)
        return resp.is_success

    async def stream_stop(self, stream_id: str) -> dict[str, Any]:
        """Stop a stream."""
        resp = await self.client.post(f"/stream/{stream_id}/stop")
        resp.raise_for_status()
        return resp.json()

    async def list_streams(self) -> list[dict[str, Any]]:
        """List active streams."""
        resp = await self.client.get("/streams")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("streams", [])
