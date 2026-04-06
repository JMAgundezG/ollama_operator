"""ollama_operator - Intermediary library between clients and the Ollama API.

The main feature is automatic model management: before forwarding a request,
the operator checks whether the requested model is available locally and
pulls it if it is not.

Quick start (sync)::

    from ollama_operator import OllamaOperator

    with OllamaOperator() as op:
        resp = op.chat("llama3.2", messages=[
            {"role": "user", "content": "Hello!"},
        ])
        print(resp.message.content)

Quick start (async)::

    import asyncio
    from ollama_operator import AsyncOllamaOperator

    async def main():
        async with AsyncOllamaOperator() as op:
            resp = await op.chat("llama3.2", messages=[
                {"role": "user", "content": "Hello!"},
            ])
            print(resp.message.content)

    asyncio.run(main())
"""

from .async_client import AsyncOllamaOperator
from .client import OllamaOperator
from .exceptions import (
    ConnectionError,
    ModelNotFoundError,
    ModelPullError,
    OllamaOperatorError,
    ResponseError,
    TimeoutError,
)
from .types import (
    ChatResponse,
    EmbedResponse,
    GenerateResponse,
    Message,
    ModelDetails,
    ModelInfo,
    ModelShowResponse,
    PullProgress,
    RunningModel,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolFunction,
    VersionResponse,
)

__all__ = [
    # Clients
    "OllamaOperator",
    "AsyncOllamaOperator",
    # Types
    "ChatResponse",
    "EmbedResponse",
    "GenerateResponse",
    "Message",
    "ModelDetails",
    "ModelInfo",
    "ModelShowResponse",
    "PullProgress",
    "RunningModel",
    "Tool",
    "ToolCall",
    "ToolCallFunction",
    "ToolFunction",
    "VersionResponse",
    # Exceptions
    "OllamaOperatorError",
    "ConnectionError",
    "ModelNotFoundError",
    "ModelPullError",
    "ResponseError",
    "TimeoutError",
]

__version__ = "0.1.0"
