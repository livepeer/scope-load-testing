"""Typed async HTTP client for the Scope API."""

from typing import Any

import httpx


class ScopeClient:
    """Async HTTP client for a single Scope instance."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ScopeClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ScopeClient not entered as async context manager")
        return self._client

    # --- Health ---

    async def health(self) -> dict[str, Any]:
        resp = await self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    # --- Cloud ---

    async def cloud_connect(
        self,
        app_id: str | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if app_id:
            body["app_id"] = app_id
        if api_key:
            body["api_key"] = api_key
        if user_id:
            body["user_id"] = user_id
        resp = await self.client.post("/api/v1/cloud/connect", json=body)
        resp.raise_for_status()
        return resp.json()

    async def cloud_status(self) -> dict[str, Any]:
        resp = await self.client.get("/api/v1/cloud/status")
        resp.raise_for_status()
        return resp.json()

    async def cloud_disconnect(self) -> dict[str, Any]:
        resp = await self.client.post("/api/v1/cloud/disconnect")
        resp.raise_for_status()
        return resp.json()

    # --- Pipeline ---

    async def pipeline_load(self, pipeline_ids: list[str]) -> dict[str, Any]:
        resp = await self.client.post(
            "/api/v1/pipeline/load", json={"pipeline_ids": pipeline_ids}
        )
        resp.raise_for_status()
        return resp.json()

    async def pipeline_status(self) -> dict[str, Any]:
        resp = await self.client.get("/api/v1/pipeline/status")
        resp.raise_for_status()
        return resp.json()

    # --- Session ---

    async def session_start(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self.client.post("/api/v1/session/start", json=body)
        resp.raise_for_status()
        return resp.json()

    async def session_stop(self) -> dict[str, Any]:
        resp = await self.client.post("/api/v1/session/stop")
        resp.raise_for_status()
        return resp.json()

    async def session_metrics(self) -> dict[str, Any]:
        resp = await self.client.get("/api/v1/session/metrics")
        resp.raise_for_status()
        return resp.json()

    async def session_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        resp = await self.client.post("/api/v1/session/parameters", json=params)
        resp.raise_for_status()
        return resp.json()

    async def capture_frame(
        self, sink_node_id: str | None = None, quality: int = 85
    ) -> bytes:
        params: dict[str, Any] = {"quality": quality}
        if sink_node_id:
            params["sink_node_id"] = sink_node_id
        resp = await self.client.get("/api/v1/session/frame", params=params)
        resp.raise_for_status()
        return resp.content

    # --- Logs ---

    async def get_logs(self, lines: int = 50) -> dict[str, Any]:
        resp = await self.client.get("/api/v1/logs/tail", params={"lines": lines})
        resp.raise_for_status()
        return resp.json()
