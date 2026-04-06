"""Unit tests for ollama_operator types."""

from ollama_operator.types import (
    ChatResponse,
    EmbedResponse,
    GenerateResponse,
    Message,
    ModelDetails,
    ModelInfo,
    ModelShowResponse,
    PullProgress,
    RunningModel,
    ToolCall,
    ToolCallFunction,
    VersionResponse,
)


class TestMessage:
    def test_basic_to_dict(self):
        msg = Message(role="user", content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}

    def test_to_dict_with_images(self):
        msg = Message(role="user", content="look", images=["abc123"])
        d = msg.to_dict()
        assert d["images"] == ["abc123"]

    def test_to_dict_with_tool_calls(self):
        tc = ToolCall(function=ToolCallFunction(name="fn", arguments={"a": 1}))
        msg = Message(role="assistant", content="", tool_calls=[tc])
        d = msg.to_dict()
        assert d["tool_calls"] == [{"function": {"name": "fn", "arguments": {"a": 1}}}]

    def test_optional_fields_omitted(self):
        msg = Message(role="user", content="hi")
        d = msg.to_dict()
        assert "images" not in d
        assert "tool_calls" not in d
        assert "thinking" not in d
        assert "tool_name" not in d


class TestGenerateResponse:
    def test_from_dict_minimal(self):
        r = GenerateResponse.from_dict({"done": True})
        assert r.done is True
        assert r.response == ""

    def test_from_dict_full(self):
        data = {
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "response": "hello world",
            "done": True,
            "done_reason": "stop",
            "context": [1, 2, 3],
            "total_duration": 5043500667,
            "load_duration": 5025959,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 325953000,
            "eval_count": 290,
            "eval_duration": 4709213000,
        }
        r = GenerateResponse.from_dict(data)
        assert r.model == "llama3.2"
        assert r.response == "hello world"
        assert r.context == [1, 2, 3]
        assert r.eval_count == 290


class TestChatResponse:
    def test_from_dict_with_message(self):
        data = {
            "model": "llama3.2",
            "created_at": "2023-12-12T14:13:43.416799Z",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "total_duration": 5191566416,
        }
        r = ChatResponse.from_dict(data)
        assert r.message is not None
        assert r.message.role == "assistant"
        assert r.message.content == "Hello!"

    def test_from_dict_with_tool_calls(self):
        data = {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Tokyo"},
                        }
                    }
                ],
            },
            "done": True,
        }
        r = ChatResponse.from_dict(data)
        assert r.message is not None
        assert r.message.tool_calls is not None
        assert len(r.message.tool_calls) == 1
        assert r.message.tool_calls[0].function.name == "get_weather"
        assert r.message.tool_calls[0].function.arguments == {"city": "Tokyo"}

    def test_from_dict_streaming_chunk(self):
        data = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "The"},
            "done": False,
        }
        r = ChatResponse.from_dict(data)
        assert r.done is False
        assert r.message.content == "The"


class TestEmbedResponse:
    def test_from_dict(self):
        data = {
            "model": "all-minilm",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "total_duration": 14143917,
            "load_duration": 1019500,
            "prompt_eval_count": 8,
        }
        r = EmbedResponse.from_dict(data)
        assert r.model == "all-minilm"
        assert len(r.embeddings) == 2
        assert r.embeddings[0] == [0.1, 0.2, 0.3]


class TestModelInfo:
    def test_from_dict(self):
        data = {
            "name": "llama3.2:latest",
            "model": "llama3.2:latest",
            "modified_at": "2025-05-04T17:37:44.706015396-07:00",
            "size": 2019393189,
            "digest": "a80c4f17acd5",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "3.2B",
                "quantization_level": "Q4_K_M",
            },
        }
        m = ModelInfo.from_dict(data)
        assert m.name == "llama3.2:latest"
        assert m.size == 2019393189
        assert m.details.family == "llama"
        assert m.details.quantization_level == "Q4_K_M"


class TestModelDetails:
    def test_from_dict_empty(self):
        d = ModelDetails.from_dict({})
        assert d.format == ""
        assert d.families == []

    def test_from_dict_full(self):
        d = ModelDetails.from_dict({
            "parent_model": "base",
            "format": "gguf",
            "family": "llama",
            "families": ["llama", "qwen2"],
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M",
        })
        assert d.family == "llama"
        assert len(d.families) == 2


class TestModelShowResponse:
    def test_from_dict(self):
        data = {
            "modelfile": "FROM llama3.2",
            "parameters": "num_keep 24",
            "template": "{{ .System }}",
            "details": {"format": "gguf", "family": "llama"},
            "model_info": {"general.architecture": "llama"},
            "capabilities": ["completion"],
        }
        r = ModelShowResponse.from_dict(data)
        assert r.modelfile == "FROM llama3.2"
        assert r.capabilities == ["completion"]
        assert r.model_info["general.architecture"] == "llama"


class TestRunningModel:
    def test_from_dict(self):
        data = {
            "name": "mistral:latest",
            "model": "mistral:latest",
            "size": 5137025024,
            "digest": "2ae6f6dd7a3d",
            "details": {"format": "gguf", "family": "llama"},
            "expires_at": "2024-06-04T14:38:31.83753-07:00",
            "size_vram": 5137025024,
        }
        r = RunningModel.from_dict(data)
        assert r.expires_at.startswith("2024")
        assert r.size_vram == 5137025024


class TestPullProgress:
    def test_from_dict(self):
        p = PullProgress.from_dict({
            "status": "pulling manifest",
            "digest": "sha256:abc",
            "total": 2142590208,
            "completed": 241970,
        })
        assert p.status == "pulling manifest"
        assert p.total == 2142590208

    def test_from_dict_minimal(self):
        p = PullProgress.from_dict({"status": "success"})
        assert p.status == "success"
        assert p.total == 0


class TestVersionResponse:
    def test_from_dict(self):
        v = VersionResponse.from_dict({"version": "0.5.1"})
        assert v.version == "0.5.1"
