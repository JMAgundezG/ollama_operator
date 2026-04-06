# ollama_operator

Intermediary library between clients and the Ollama API with automatic model management.

Before forwarding any request, `ollama_operator` checks whether the requested model is available locally. If it is not, the model is pulled automatically before executing the call.

## Installation

```bash
pip install -e .
```

## Quick start

### Synchronous

```python
from ollama_operator import OllamaOperator

with OllamaOperator() as op:
    response = op.chat("llama3.2", messages=[
        {"role": "user", "content": "Why is the sky blue?"},
    ])
    print(response.message.content)
```

### Asynchronous

```python
import asyncio
from ollama_operator import AsyncOllamaOperator

async def main():
    async with AsyncOllamaOperator() as op:
        response = await op.chat("llama3.2", messages=[
            {"role": "user", "content": "Why is the sky blue?"},
        ])
        print(response.message.content)

asyncio.run(main())
```

## Features

- **Automatic model pull** &mdash; models are downloaded on-the-fly when not present locally. Disable with `auto_pull=False`.
- **Full Ollama API coverage** &mdash; `generate`, `chat`, `embed`, `list_models`, `show`, `copy`, `delete`, `pull`, `push`, `create`, `list_running`, `version`.
- **Sync and async clients** &mdash; `OllamaOperator` and `AsyncOllamaOperator` with the same interface.
- **Streaming** &mdash; token-by-token streaming for `generate` and `chat` via `stream=True`.
- **Tool calling** &mdash; first-class support through `Tool`, `ToolCall` and related types.
- **Progress callbacks** &mdash; monitor model downloads with `on_pull_progress`.

## API overview

### Generate

```python
response = op.generate("llama3.2", prompt="Explain gravity")
print(response.response)
```

### Chat

```python
response = op.chat("llama3.2", messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
])
print(response.message.content)
```

### Streaming

```python
for chunk in op.chat("llama3.2", messages=[...], stream=True):
    print(chunk.message.content, end="", flush=True)
```

### Embeddings

```python
response = op.embed("all-minilm", input="Some text to embed")
print(response.embeddings)
```

### Model management

```python
models = op.list_models()          # list local models
info = op.show("llama3.2")         # model details
op.pull("mistral")                 # explicit pull
op.copy("llama3.2", "my-backup")   # copy a model
op.delete("my-backup")             # delete a model
running = op.list_running()        # models loaded in memory
```

### Pull progress

```python
def on_progress(p):
    if p.total:
        pct = p.completed / p.total * 100
        print(f"{p.status}: {pct:.1f}%")

op = OllamaOperator(on_pull_progress=on_progress)
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `base_url` | `http://localhost:11434` | Ollama server URL |
| `auto_pull` | `True` | Pull missing models automatically |
| `on_pull_progress` | `None` | Callback for download progress |

## Running tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
