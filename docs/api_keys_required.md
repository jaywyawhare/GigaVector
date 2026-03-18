# API Keys

## All Keys

| Key | Purpose | Required? | Format | Where to Get It |
|-----|---------|-----------|--------|-----------------|
| `OPENAI_API_KEY` | OpenAI LLM and embeddings | **Required** for tests | `sk-...` | https://platform.openai.com/api-keys |
| `ANTHROPIC_API_KEY` | Anthropic/Claude LLM | **Required** for tests | `sk-ant-...` | https://console.anthropic.com/settings/keys |
| `GOOGLE_API_KEY` | Google embeddings (default model: `text-embedding-004`, dim 768) | Optional | No specific prefix | https://aistudio.google.com |
| Azure OpenAI `api_key` + `base_url` | Azure OpenAI LLM | Optional, set in config (not env vars) | 32+ alphanumeric chars | Azure Portal |
| Custom provider `api_key` + `base_url` | OpenAI-compatible APIs (Groq, Together AI, etc.) | Optional, set in config (not env vars) | Varies | Provider dashboard |
| *(none)* | HuggingFace local embeddings (`all-MiniLM-L6-v2`, etc.) | Free, no key needed | N/A | N/A |

## .env Template

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Optional
GOOGLE_API_KEY=your-google-api-key-here

# Azure OpenAI and custom providers use base_url + api_key in code/config,
# not environment variables.
```

## Security Notes

1. **Never commit `.env` files** -- ensure `.env` is in `.gitignore`.
2. **Use environment variables** in production, not `.env` files.
3. **Rotate keys regularly** and use separate keys for dev vs. production.
4. **Restrict API key permissions** when possible.
