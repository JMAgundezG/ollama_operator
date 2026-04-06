"""Type definitions for Ollama API requests and responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Shared / common types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    images: list[str] | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images is not None:
            d["images"] = self.images
        if self.thinking is not None:
            d["thinking"] = self.thinking
        if self.tool_calls is not None:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        return d


@dataclass
class ToolCall:
    """A tool/function call made by the model."""

    function: ToolCallFunction

    def to_dict(self) -> dict[str, Any]:
        return {"function": self.function.to_dict()}


@dataclass
class ToolCallFunction:
    """Function details inside a tool call."""

    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class Tool:
    """A tool definition for function calling."""

    type: str = "function"
    function: ToolFunction | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.function is not None:
            d["function"] = self.function.to_dict()
        return d


@dataclass
class ToolFunction:
    """Function definition inside a tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ModelDetails:
    """Model detail information returned by list/show endpoints."""

    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: list[str] = field(default_factory=list)
    parameter_size: str = ""
    quantization_level: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelDetails:
        return cls(
            parent_model=data.get("parent_model", ""),
            format=data.get("format", ""),
            family=data.get("family", ""),
            families=data.get("families") or [],
            parameter_size=data.get("parameter_size", ""),
            quantization_level=data.get("quantization_level", ""),
        )


# ---------------------------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------------------------

@dataclass
class GenerateResponse:
    """Response from POST /api/generate."""

    model: str = ""
    created_at: str = ""
    response: str = ""
    done: bool = False
    done_reason: str = ""
    context: list[int] = field(default_factory=list)
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0
    # Image generation fields
    image: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerateResponse:
        return cls(
            model=data.get("model", ""),
            created_at=data.get("created_at", ""),
            response=data.get("response", ""),
            done=data.get("done", False),
            done_reason=data.get("done_reason", ""),
            context=data.get("context") or [],
            total_duration=data.get("total_duration", 0),
            load_duration=data.get("load_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            prompt_eval_duration=data.get("prompt_eval_duration", 0),
            eval_count=data.get("eval_count", 0),
            eval_duration=data.get("eval_duration", 0),
            image=data.get("image", ""),
        )


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """Response from POST /api/chat."""

    model: str = ""
    created_at: str = ""
    message: Message | None = None
    done: bool = False
    done_reason: str = ""
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatResponse:
        msg_data = data.get("message")
        message = None
        if msg_data:
            tool_calls = None
            if msg_data.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        function=ToolCallFunction(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        )
                    )
                    for tc in msg_data["tool_calls"]
                ]
            message = Message(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content", ""),
                images=msg_data.get("images"),
                thinking=msg_data.get("thinking"),
                tool_calls=tool_calls,
                tool_name=msg_data.get("tool_name"),
            )
        return cls(
            model=data.get("model", ""),
            created_at=data.get("created_at", ""),
            message=message,
            done=data.get("done", False),
            done_reason=data.get("done_reason", ""),
            total_duration=data.get("total_duration", 0),
            load_duration=data.get("load_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            prompt_eval_duration=data.get("prompt_eval_duration", 0),
            eval_count=data.get("eval_count", 0),
            eval_duration=data.get("eval_duration", 0),
        )


# ---------------------------------------------------------------------------
# Embeddings endpoint
# ---------------------------------------------------------------------------

@dataclass
class EmbedResponse:
    """Response from POST /api/embed."""

    model: str = ""
    embeddings: list[list[float]] = field(default_factory=list)
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbedResponse:
        return cls(
            model=data.get("model", ""),
            embeddings=data.get("embeddings") or [],
            total_duration=data.get("total_duration", 0),
            load_duration=data.get("load_duration", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
        )


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """A model entry returned by GET /api/tags."""

    name: str = ""
    model: str = ""
    modified_at: str = ""
    size: int = 0
    digest: str = ""
    details: ModelDetails = field(default_factory=ModelDetails)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        return cls(
            name=data.get("name", ""),
            model=data.get("model", ""),
            modified_at=data.get("modified_at", ""),
            size=data.get("size", 0),
            digest=data.get("digest", ""),
            details=ModelDetails.from_dict(data.get("details") or {}),
        )


@dataclass
class ModelShowResponse:
    """Response from POST /api/show."""

    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: ModelDetails = field(default_factory=ModelDetails)
    model_info: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelShowResponse:
        return cls(
            modelfile=data.get("modelfile", ""),
            parameters=data.get("parameters", ""),
            template=data.get("template", ""),
            details=ModelDetails.from_dict(data.get("details") or {}),
            model_info=data.get("model_info") or {},
            capabilities=data.get("capabilities") or [],
        )


@dataclass
class RunningModel:
    """A running model entry returned by GET /api/ps."""

    name: str = ""
    model: str = ""
    size: int = 0
    digest: str = ""
    details: ModelDetails = field(default_factory=ModelDetails)
    expires_at: str = ""
    size_vram: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunningModel:
        return cls(
            name=data.get("name", ""),
            model=data.get("model", ""),
            size=data.get("size", 0),
            digest=data.get("digest", ""),
            details=ModelDetails.from_dict(data.get("details") or {}),
            expires_at=data.get("expires_at", ""),
            size_vram=data.get("size_vram", 0),
        )


@dataclass
class PullProgress:
    """A single progress event when pulling a model."""

    status: str = ""
    digest: str = ""
    total: int = 0
    completed: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PullProgress:
        return cls(
            status=data.get("status", ""),
            digest=data.get("digest", ""),
            total=data.get("total", 0),
            completed=data.get("completed", 0),
        )


@dataclass
class VersionResponse:
    """Response from GET /api/version."""

    version: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionResponse:
        return cls(version=data.get("version", ""))
