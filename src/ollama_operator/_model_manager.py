"""Model verification and automatic installation logic.

This is the core differentiator of ``ollama_operator``: before forwarding a
request to Ollama, it checks whether the requested model is available locally
and, if not, pulls it automatically.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ._http import AsyncHTTP, SyncHTTP
from .exceptions import ModelPullError, ResponseError
from .types import PullProgress

logger = logging.getLogger("ollama_operator")

# Type alias for the optional user-supplied progress callback.
ProgressCallback = Callable[[PullProgress], None] | None


def _model_names(models_response: dict[str, Any]) -> set[str]:
    """Extract a set of model names from a ``/api/tags`` response."""
    names: set[str] = set()
    for m in models_response.get("models", []):
        name = m.get("name", "")
        if name:
            names.add(name)
            # Also add without the `:latest` tag so callers can use short names.
            if ":" in name:
                names.add(name.split(":")[0])
    return names


def _normalise_model(model: str) -> str:
    """Ensure the model name has an explicit tag."""
    if ":" not in model:
        return f"{model}:latest"
    return model


# ---------------------------------------------------------------------------
# Synchronous model manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Synchronous model verifier / auto-puller."""

    def __init__(self, http: SyncHTTP) -> None:
        self._http = http
        self._known_models: set[str] = set()

    def refresh(self) -> set[str]:
        """Fetch the list of locally available models and cache them."""
        data = self._http.get("/api/tags")
        self._known_models = _model_names(data)
        return self._known_models

    def is_available(self, model: str) -> bool:
        """Return *True* if *model* is installed locally."""
        if not self._known_models:
            self.refresh()
        normalised = _normalise_model(model)
        return model in self._known_models or normalised in self._known_models

    def ensure(
        self,
        model: str,
        *,
        on_progress: ProgressCallback = None,
    ) -> None:
        """Make sure *model* is available, pulling it if necessary.

        Parameters
        ----------
        model:
            The model name (e.g. ``"llama3.2"`` or ``"llama3.2:latest"``).
        on_progress:
            Optional callback invoked with :class:`PullProgress` objects
            while the model is being downloaded.
        """
        if self.is_available(model):
            logger.debug("Model '%s' is already available locally.", model)
            return

        logger.info("Model '%s' not found locally. Pulling...", model)
        self._pull(model, on_progress=on_progress)
        # Refresh cache after pull.
        self.refresh()

    def _pull(
        self,
        model: str,
        *,
        on_progress: ProgressCallback = None,
    ) -> None:
        payload = {"model": model, "stream": True}
        last_status = ""
        try:
            for chunk in self._http.post_stream("/api/pull", payload):
                progress = PullProgress.from_dict(chunk)
                last_status = progress.status
                if on_progress is not None:
                    on_progress(progress)
                logger.debug("Pull %s: %s", model, progress.status)
        except ResponseError as exc:
            raise ModelPullError(model, str(exc)) from exc

        if last_status != "success":
            raise ModelPullError(model, f"Unexpected final status: {last_status}")


# ---------------------------------------------------------------------------
# Asynchronous model manager
# ---------------------------------------------------------------------------

class AsyncModelManager:
    """Asynchronous model verifier / auto-puller."""

    def __init__(self, http: AsyncHTTP) -> None:
        self._http = http
        self._known_models: set[str] = set()

    async def refresh(self) -> set[str]:
        data = await self._http.get("/api/tags")
        self._known_models = _model_names(data)
        return self._known_models

    async def is_available(self, model: str) -> bool:
        if not self._known_models:
            await self.refresh()
        normalised = _normalise_model(model)
        return model in self._known_models or normalised in self._known_models

    async def ensure(
        self,
        model: str,
        *,
        on_progress: ProgressCallback = None,
    ) -> None:
        if await self.is_available(model):
            logger.debug("Model '%s' is already available locally.", model)
            return

        logger.info("Model '%s' not found locally. Pulling...", model)
        await self._pull(model, on_progress=on_progress)
        await self.refresh()

    async def _pull(
        self,
        model: str,
        *,
        on_progress: ProgressCallback = None,
    ) -> None:
        payload = {"model": model, "stream": True}
        last_status = ""
        try:
            async for chunk in self._http.post_stream("/api/pull", payload):
                progress = PullProgress.from_dict(chunk)
                last_status = progress.status
                if on_progress is not None:
                    on_progress(progress)
                logger.debug("Pull %s: %s", model, progress.status)
        except ResponseError as exc:
            raise ModelPullError(model, str(exc)) from exc

        if last_status != "success":
            raise ModelPullError(model, f"Unexpected final status: {last_status}")
