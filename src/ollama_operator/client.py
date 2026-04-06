"""Synchronous Ollama Operator client.

Usage::

    from ollama_operator import OllamaOperator

    with OllamaOperator() as op:
        # The model will be pulled automatically if not installed.
        response = op.chat("llama3.2", messages=[
            {"role": "user", "content": "Hello!"},
        ])
        print(response.message.content)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from ._http import SyncHTTP
from ._model_manager import ModelManager, ProgressCallback
from .types import (
    ChatResponse,
    EmbedResponse,
    GenerateResponse,
    Message,
    ModelInfo,
    ModelShowResponse,
    PullProgress,
    RunningModel,
    VersionResponse,
)

logger = logging.getLogger("ollama_operator")


class OllamaOperator:
    """Synchronous client that proxies Ollama API requests with automatic
    model management.

    Parameters
    ----------
    base_url:
        Base URL of the Ollama server (default ``http://localhost:11434``).
    auto_pull:
        When *True* (the default), any request that references a model not
        yet installed locally will trigger an automatic ``pull`` before the
        actual API call.
    on_pull_progress:
        Optional callback invoked with :class:`~ollama_operator.types.PullProgress`
        objects while a model is being downloaded.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        auto_pull: bool = True,
        on_pull_progress: ProgressCallback = None,
    ) -> None:
        self._http = SyncHTTP(base_url=base_url)
        self._models = ModelManager(self._http)
        self.auto_pull = auto_pull
        self.on_pull_progress = on_pull_progress

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Release underlying HTTP resources."""
        self._http.close()

    def __enter__(self) -> OllamaOperator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- private helpers -----------------------------------------------------

    def _ensure_model(self, model: str) -> None:
        if self.auto_pull:
            self._models.ensure(model, on_progress=self.on_pull_progress)

    @staticmethod
    def _to_message_dicts(
        messages: list[Message | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            m.to_dict() if isinstance(m, Message) else m for m in messages
        ]

    # -- Generate endpoint ---------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str = "",
        *,
        suffix: str | None = None,
        images: list[str] | None = None,
        format: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        system: str | None = None,
        template: str | None = None,
        raw: bool = False,
        keep_alive: str | int | None = None,
        think: bool | None = None,
        stream: bool = False,
    ) -> GenerateResponse | Iterator[GenerateResponse]:
        """POST /api/generate"""
        self._ensure_model(model)

        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": stream}
        if suffix is not None:
            payload["suffix"] = suffix
        if images is not None:
            payload["images"] = images
        if format is not None:
            payload["format"] = format
        if options is not None:
            payload["options"] = options
        if system is not None:
            payload["system"] = system
        if template is not None:
            payload["template"] = template
        if raw:
            payload["raw"] = True
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if think is not None:
            payload["think"] = think

        if stream:
            return self._generate_stream(payload)

        data = self._http.post("/api/generate", payload)
        return GenerateResponse.from_dict(data)

    def _generate_stream(
        self, payload: dict[str, Any]
    ) -> Iterator[GenerateResponse]:
        for chunk in self._http.post_stream("/api/generate", payload):
            yield GenerateResponse.from_dict(chunk)

    # -- Chat endpoint -------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[Message | dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        format: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
        think: bool | None = None,
        stream: bool = False,
    ) -> ChatResponse | Iterator[ChatResponse]:
        """POST /api/chat"""
        self._ensure_model(model)

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._to_message_dicts(messages),
            "stream": stream,
        }
        if tools is not None:
            payload["tools"] = tools
        if format is not None:
            payload["format"] = format
        if options is not None:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if think is not None:
            payload["think"] = think

        if stream:
            return self._chat_stream(payload)

        data = self._http.post("/api/chat", payload)
        return ChatResponse.from_dict(data)

    def _chat_stream(
        self, payload: dict[str, Any]
    ) -> Iterator[ChatResponse]:
        for chunk in self._http.post_stream("/api/chat", payload):
            yield ChatResponse.from_dict(chunk)

    # -- Embed endpoint ------------------------------------------------------

    def embed(
        self,
        model: str,
        input: str | list[str],
        *,
        truncate: bool | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> EmbedResponse:
        """POST /api/embed"""
        self._ensure_model(model)

        payload: dict[str, Any] = {"model": model, "input": input}
        if truncate is not None:
            payload["truncate"] = truncate
        if options is not None:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        data = self._http.post("/api/embed", payload)
        return EmbedResponse.from_dict(data)

    # -- Model management endpoints ------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """GET /api/tags - List locally available models."""
        data = self._http.get("/api/tags")
        return [ModelInfo.from_dict(m) for m in data.get("models", [])]

    def show(self, model: str, *, verbose: bool = False) -> ModelShowResponse:
        """POST /api/show - Show model information."""
        payload: dict[str, Any] = {"model": model}
        if verbose:
            payload["verbose"] = True
        data = self._http.post("/api/show", payload)
        return ModelShowResponse.from_dict(data)

    def copy(self, source: str, destination: str) -> None:
        """POST /api/copy - Copy a model."""
        self._http.post("/api/copy", {"source": source, "destination": destination})

    def delete(self, model: str) -> None:
        """DELETE /api/delete - Delete a model."""
        self._http.delete("/api/delete", {"model": model})
        # Invalidate cache.
        self._models.refresh()

    def pull(
        self,
        model: str,
        *,
        insecure: bool = False,
        on_progress: ProgressCallback = None,
    ) -> None:
        """POST /api/pull - Explicitly pull a model.

        This bypasses the ``auto_pull`` check and always downloads.
        """
        payload: dict[str, Any] = {"model": model, "stream": True}
        if insecure:
            payload["insecure"] = True
        for chunk in self._http.post_stream("/api/pull", payload):
            if on_progress is not None:
                on_progress(PullProgress.from_dict(chunk))
        self._models.refresh()

    def push(
        self,
        model: str,
        *,
        insecure: bool = False,
        on_progress: ProgressCallback = None,
    ) -> None:
        """POST /api/push - Push a model to a registry."""
        payload: dict[str, Any] = {"model": model, "stream": True}
        if insecure:
            payload["insecure"] = True
        for chunk in self._http.post_stream("/api/push", payload):
            if on_progress is not None:
                on_progress(PullProgress.from_dict(chunk))

    def create(
        self,
        model: str,
        *,
        from_model: str | None = None,
        files: dict[str, str] | None = None,
        adapters: dict[str, str] | None = None,
        template: str | None = None,
        license: str | list[str] | None = None,
        system: str | None = None,
        parameters: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        quantize: str | None = None,
        stream: bool = True,
        on_progress: ProgressCallback = None,
    ) -> None:
        """POST /api/create - Create a model."""
        payload: dict[str, Any] = {"model": model, "stream": stream}
        if from_model is not None:
            payload["from"] = from_model
        if files is not None:
            payload["files"] = files
        if adapters is not None:
            payload["adapters"] = adapters
        if template is not None:
            payload["template"] = template
        if license is not None:
            payload["license"] = license
        if system is not None:
            payload["system"] = system
        if parameters is not None:
            payload["parameters"] = parameters
        if messages is not None:
            payload["messages"] = messages
        if quantize is not None:
            payload["quantize"] = quantize

        if stream:
            for chunk in self._http.post_stream("/api/create", payload):
                if on_progress is not None:
                    on_progress(PullProgress.from_dict(chunk))
        else:
            self._http.post("/api/create", payload)
        self._models.refresh()

    # -- Running / PS endpoint -----------------------------------------------

    def list_running(self) -> list[RunningModel]:
        """GET /api/ps - List currently running models."""
        data = self._http.get("/api/ps")
        return [RunningModel.from_dict(m) for m in data.get("models", [])]

    # -- Version endpoint ----------------------------------------------------

    def version(self) -> VersionResponse:
        """GET /api/version"""
        data = self._http.get("/api/version")
        return VersionResponse.from_dict(data)
