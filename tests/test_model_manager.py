"""Tests for the ModelManager and AsyncModelManager."""

import httpx
import pytest
import respx

from ollama_operator._model_manager import ModelManager, _model_names, _normalise_model
from ollama_operator._http import SyncHTTP
from ollama_operator.exceptions import ModelPullError


BASE_URL = "http://localhost:11434"


class TestHelpers:
    def test_model_names_extracts_names(self):
        data = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:7b"},
            ]
        }
        names = _model_names(data)
        assert "llama3.2:latest" in names
        assert "llama3.2" in names  # short name
        assert "mistral:7b" in names
        assert "mistral" in names  # short name

    def test_model_names_empty(self):
        assert _model_names({"models": []}) == set()
        assert _model_names({}) == set()

    def test_normalise_model_adds_latest(self):
        assert _normalise_model("llama3.2") == "llama3.2:latest"

    def test_normalise_model_keeps_tag(self):
        assert _normalise_model("llama3.2:7b") == "llama3.2:7b"


class TestModelManager:
    @respx.mock
    def test_refresh(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={
                "models": [{"name": "llama3.2:latest"}]
            })
        )

        http = SyncHTTP(base_url=BASE_URL)
        mm = ModelManager(http)
        names = mm.refresh()

        assert "llama3.2:latest" in names
        assert "llama3.2" in names
        http.close()

    @respx.mock
    def test_is_available_true(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={
                "models": [{"name": "llama3.2:latest"}]
            })
        )

        http = SyncHTTP(base_url=BASE_URL)
        mm = ModelManager(http)

        assert mm.is_available("llama3.2") is True
        assert mm.is_available("llama3.2:latest") is True
        http.close()

    @respx.mock
    def test_is_available_false(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={
                "models": [{"name": "llama3.2:latest"}]
            })
        )

        http = SyncHTTP(base_url=BASE_URL)
        mm = ModelManager(http)

        assert mm.is_available("nonexistent") is False
        http.close()

    @respx.mock
    def test_ensure_skips_when_available(self):
        respx.get(f"{BASE_URL}/api/tags").mock(
            return_value=httpx.Response(200, json={
                "models": [{"name": "llama3.2:latest"}]
            })
        )

        http = SyncHTTP(base_url=BASE_URL)
        mm = ModelManager(http)
        mm.ensure("llama3.2")  # Should not raise or call pull

        assert not any(
            call.request.url.path == "/api/pull"
            for call in respx.calls
        )
        http.close()
