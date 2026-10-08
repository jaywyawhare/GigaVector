# API Keys Required

GigaVector integrates with external LLM and embedding providers. API keys are passed in code/config at runtime - environment variables are used by tests and optional dashboard fallbacks.

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required for | Used by |
|----------|--------------|---------|
| `OPENAI_API_KEY` | OpenAI LLM/embedding tests | `test_llm.c`, `test_memory_llm.c`, `test_embedding.c` |
| `ANTHROPIC_API_KEY` | Anthropic LLM tests | `test_llm.c` |
| `GOOGLE_API_KEY` | Google embedding + Gemini LLM tests | `test_embedding.c`, `test_llm.c`, `test_auto_embed.c`, `test_inference.c` |
| `GEMINI_API_KEY` | Legacy alias for `GOOGLE_API_KEY` | Same tests (fallback) |
| `GV_WAL_DIR` | Optional WAL override | Database WAL location |

## Google / Gemini

Get a key from [Google AI Studio](https://aistudio.google.com).

| Use case | Provider | Default model | Default dimension |
|----------|----------|---------------|-------------------|
| Embeddings | `EmbeddingProvider.GOOGLE` / `AutoEmbedProvider.GOOGLE` | `text-embedding-004` | 768 |
| LLM chat | `LLMProvider.GOOGLE` | `gemini-2.5-flash` | - |
| Inference | `embed_provider="google"` | `text-embedding-004` | 768 |
| Agents | `llm_provider="google"` | `gemini-2.5-flash` | - |

### Python example

```python
import os
from gigavector import AutoEmbedConfig, AutoEmbedProvider, AutoEmbedder

embedder = AutoEmbedder(AutoEmbedConfig(
    provider=AutoEmbedProvider.GOOGLE,
    api_key=os.environ["GOOGLE_API_KEY"],
    model_name="text-embedding-004",
    dimension=768,
))
```

## Azure OpenAI and Custom Providers

Azure OpenAI and custom OpenAI-compatible endpoints require `base_url` and `api_key` in code/config - not environment variables.

## CI

GitHub Actions CI does not set provider API keys. Live API tests are skipped automatically when keys are absent.
