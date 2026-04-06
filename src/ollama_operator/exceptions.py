"""Custom exceptions for ollama_operator."""

from __future__ import annotations


class OllamaOperatorError(Exception):
    """Base exception for all ollama_operator errors."""


class ConnectionError(OllamaOperatorError):
    """Raised when a connection to the Ollama server cannot be established."""


class ModelNotFoundError(OllamaOperatorError):
    """Raised when a requested model is not found locally or remotely."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model '{model}' not found")


class ModelPullError(OllamaOperatorError):
    """Raised when pulling a model from the registry fails."""

    def __init__(self, model: str, reason: str = "") -> None:
        self.model = model
        self.reason = reason
        msg = f"Failed to pull model '{model}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ResponseError(OllamaOperatorError):
    """Raised when the Ollama API returns an error response."""

    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class TimeoutError(OllamaOperatorError):
    """Raised when a request to the Ollama API times out."""
