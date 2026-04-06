"""Tests for the synchronous OllamaOperator client using respx mocks."""

import json

import httpx
import pytest
import respx

from ollama_operator import OllamaOperator
from ollama_operator.exceptions import ModelPullError, ResponseError


BASE_URL = "http://localhost:11434"

# Reusable fixtures -----------------------------------------------------------

TAGS_RESPONSE = {
    "models": [
        {
            "name": "llama3.2:latest",
            "model": "llama3.2:latest",
            "modified_at": "2025-05-04T17:37:44Z",
            "size": 2019393189,
            "digest": "abc123",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "3.2B",
                "quantization_level": "Q4_K_M",
            },
        }
    ]
}


def _ndjson(*objects: dict) -> str:
    """Encode multiple dicts as newline-delimited JSON."""
    return "\n".join(json.dumps(o) for o in objects)


# Tests -----------------------------------------------------------------------


class TestModelAutoDetection:
    @respx.mock
    def test_no_pull_when_model_exists(self):
        """When the model is already local, no pull should happen."""
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=TAGS_RESPONSE)
        )
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json={
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hi!"},
                "done": True,
            })
        )

        with OllamaOperator(base_url=BASE_URL) as op:
            resp = op.chat("llama3.2", messages=[
                {"role": "user", "content": "hello"},
            ])

        assert resp.message.content == "Hi!"
        # /api/pull should NOT have been called
        assert not any(
            call.request.url.path == "/api/pull"
            for call in respx.calls
        )

    @respx.mock
    def test_auto_pull_when_model_missing(self):
        """When the model is missing, auto_pull triggers /api/pull."""
        # First call to /api/tags returns empty -> model not found
        tags_route = respx.get(f"{BASE_URL}/api/tags")
        tags_route.side_effect = [
            httpx.Response(200, json={"models": []}),
            # After pull, refresh returns the model
            httpx.Response(200, json=TAGS_RESPONSE),
        ]

        # Simulate pull streaming response
        pull_body = _ndjson(
            {"status": "pulling manifest"},
            {"status": "success"},
        )
        respx.post(f"{BASE_URL}/api/pull").mock(
            return_value=httpx.Response(
                200,
                content=pull_body.encode(),
                headers={"content-type": "application/x-ndjson"},
            )
        )

        respx.post(f"{BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={
                "model": "llama3.2",
                "response": "Blue sky",
                "done": True,
            })
        )

        with OllamaOperator(base_url=BASE_URL) as op:
            resp = op.generate("llama3.2", prompt="Why is the sky blue?")

        assert resp.response == "Blue sky"
        # Pull should have been called
        assert any(
            call.request.url.path == "/api/pull"
            for call in respx.calls
        )

    @respx.mock
    def test_auto_pull_disabled(self):
        """When auto_pull=False, missing models cause direct API call."""
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json={
                "model": "unknown-model",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
            })
        )

        with OllamaOperator(base_url=BASE_URL, auto_pull=False) as op:
            resp = op.chat("unknown-model", messages=[
                {"role": "user", "content": "hi"},
            ])

        assert resp.message.content == "ok"


class TestGenerate:
    @respx.mock
    def test_generate_non_streaming(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=TAGS_RESPONSE)
        )
        respx.post(f"{BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={
                "model": "llama3.2",
                "response": "42",
                "done": True,
                "eval_count": 10,
            })
        )

        with OllamaOperator(base_url=BASE_URL) as op:
            resp = op.generate("llama3.2", prompt="Answer")

        assert resp.response == "42"
        assert resp.eval_count == 10


class TestChat:
    @respx.mock
    def test_chat_non_streaming(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=TAGS_RESPONSE)
        )
        respx.post(f"{BASE_URL}/api/chat").mock(
            return_value=httpx.Response(200, json={
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "I'm fine!"},
                "done": True,
            })
        )

        with OllamaOperator(base_url=BASE_URL) as op:
            resp = op.chat("llama3.2", messages=[
                {"role": "user", "content": "How are you?"},
            ])

        assert resp.message.content == "I'm fine!"


class TestEmbed:
    @respx.mock
    def test_embed(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=TAGS_RESPONSE)
        )
        respx.post(f"{BASE_URL}/api/embed").mock(
            return_value=httpx.Response(200, json={
                "model": "all-minilm",
                "embeddings": [[0.1, 0.2, 0.3]],
                "total_duration": 14143917,
            })
        )

        # Use auto_pull=False since we're testing embed, not model resolution
        with OllamaOperator(base_url=BASE_URL, auto_pull=False) as op:
            resp = op.embed("all-minilm", input="hello")

        assert len(resp.embeddings) == 1
        assert resp.embeddings[0] == [0.1, 0.2, 0.3]


class TestModelManagement:
    @respx.mock
    def test_list_models(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json=TAGS_RESPONSE)
        )

        with OllamaOperator(base_url=BASE_URL, auto_pull=False) as op:
            models = op.list_models()

        assert len(models) == 1
        assert models[0].name == "llama3.2:latest"

    @respx.mock
    def test_show_model(self):
        respx.post(f"{BASE_URL}/api/show").mock(
            return_value=httpx.Response(200, json={
                "modelfile": "FROM llama3.2",
                "parameters": "num_keep 24",
                "template": "{{ .System }}",
                "details": {"format": "gguf", "family": "llama"},
                "model_info": {},
                "capabilities": ["completion"],
            })
        )

        with OllamaOperator(base_url=BASE_URL, auto_pull=False) as op:
            info = op.show("llama3.2")

        assert info.modelfile == "FROM llama3.2"
        assert "completion" in info.capabilities

    @respx.mock
    def test_version(self):
        respx.get(f"{BASE_URL}/api/version").mock(
            return_value=httpx.Response(200, json={"version": "0.5.1"})
        )

        with OllamaOperator(base_url=BASE_URL, auto_pull=False) as op:
            v = op.version()

        assert v.version == "0.5.1"

    @respx.mock
    def test_list_running(self):
        respx.get(f"{BASE_URL}/api/ps").mock(
            return_value=httpx.Response(200, json={
                "models": [
                    {
                        "name": "llama3.2:latest",
                        "model": "llama3.2:latest",
                        "size": 5137025024,
                        "digest": "abc",
                        "details": {"format": "gguf", "family": "llama"},
                        "expires_at": "2024-06-04T14:38:31.83753-07:00",
                        "size_vram": 5137025024,
                    }
                ]
            })
        )

        with OllamaOperator(base_url=BASE_URL, auto_pull=False) as op:
            running = op.list_running()

        assert len(running) == 1
        assert running[0].size_vram == 5137025024


class TestErrorHandling:
    @respx.mock
    def test_response_error(self):
        respx.post(f"{BASE_URL}/api/generate").mock(
            return_value=httpx.Response(404, json={"error": "model not found"})
        )

        with OllamaOperator(base_url=BASE_URL, auto_pull=False) as op:
            with pytest.raises(ResponseError) as exc_info:
                op.generate("nonexistent", prompt="hi")

        assert exc_info.value.status_code == 404
