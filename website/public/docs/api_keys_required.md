# API Keys Required - Complete Reference

This document lists all API keys used in GigaVector, categorized by requirement level and feature.

## Quick Reference

### Required for Testing
- `OPENAI_API_KEY` - For LLM and embedding tests
- `ANTHROPIC_API_KEY` - For Anthropic/Claude LLM tests

### Optional (Feature-Specific)
- `GOOGLE_API_KEY` - For Google embeddings
- Azure OpenAI - Requires `base_url` + `api_key` (no env var name)

---

## Required API Keys

### For Running Tests

#### 1. `OPENAI_API_KEY`
- **Purpose**: OpenAI LLM and embedding services
- **Used in**:
  - `tests/test_llm.c` - LLM API call tests
  - `tests/test_memory_llm.c` - Memory extraction tests
  - Embedding service (if using OpenAI embeddings)
- **Format**: Must start with `sk-`
- **Get it**: https://platform.openai.com/api-keys
- **Example**: `OPENAI_API_KEY=sk-proj-...`
- **Status**: **REQUIRED** for real API call tests

#### 2. `ANTHROPIC_API_KEY`
- **Purpose**: Anthropic/Claude LLM services
- **Used in**:
  - `tests/test_llm.c` - Anthropic API call tests
- **Format**: Must start with `sk-ant-`
- **Get it**: https://console.anthropic.com/settings/keys
- **Example**: `ANTHROPIC_API_KEY=sk-ant-...`
- **Status**: **REQUIRED** for Anthropic LLM tests

---

## Optional API Keys

### For Embedding Services

#### 3. `GOOGLE_API_KEY`
- **Purpose**: Google Generative AI embeddings
- **Used in**:
  - `src/gv_embedding.c` - Google embedding service
- **Format**: Google API key (no specific prefix)
- **Get it**: https://aistudio.google.com
- **Example**: `GOOGLE_API_KEY=your-google-api-key`
- **Status**: **OPTIONAL** - Only needed for Google embeddings
- **Default Model**: `text-embedding-004`
- **Default Dimension**: 768

### For LLM Services (Not in Tests)

#### 4. Azure OpenAI
- **Purpose**: Azure OpenAI LLM services
- **Configuration**: 
  - Requires `base_url` (not an env var, set in config)
  - Requires `api_key` (not an env var, set in config)
- **Format**: 32+ alphanumeric characters
- **Get it**: Azure Portal → Your OpenAI resource
- **Status**: **OPTIONAL** - Only needed for Azure OpenAI
- **Note**: Must provide both `base_url` and `api_key` in code/config

#### 5. Custom LLM Providers
- **Purpose**: Custom/OpenAI-compatible APIs
- **Configuration**: 
  - Requires `base_url` (set in config)
  - Requires `api_key` (set in config)
- **Status**: **OPTIONAL** - For custom endpoints
- **Examples**: Groq, Together AI, LiteLLM, etc.

---

## No API Key Required

### Local/Free Services

#### HuggingFace Embeddings
- **Purpose**: Local embedding models
- **API Key**: Not required
- **Status**: **FREE** - Runs locally
- **Models**: Any sentence-transformers model
- **Examples**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`

---

## Complete .env File Template

Create a `.env` file in the project root with:

```bash
# OpenAI API Key (Required for LLM and embedding tests)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-api-key-here

# Anthropic API Key (Required for Anthropic LLM tests)
# Get from: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Google API Key (Optional - for Google embeddings)
# Get from: https://aistudio.google.com
GOOGLE_API_KEY=your-google-api-key-here

# Note: Azure OpenAI and Custom providers
# require base_url + api_key in code/config,
# not environment variables
```

---

## Usage by Feature

### LLM Features

| Feature | Required Keys | Optional Keys |
|---------|--------------|---------------|
| OpenAI LLM | `OPENAI_API_KEY` | - |
| Anthropic LLM | `ANTHROPIC_API_KEY` | - |
| Azure OpenAI | - | `api_key` + `base_url` (in config) |
| Custom LLM | - | `api_key` + `base_url` (in config) |

### Embedding Features

| Feature | Required Keys | Optional Keys |
|---------|--------------|---------------|
| OpenAI Embeddings | `OPENAI_API_KEY` | - |
| Google Embeddings | - | `GOOGLE_API_KEY` |
| HuggingFace Embeddings | None (local) | - |
| Custom Embeddings | - | `api_key` + `base_url` (in config) |

### Testing

| Test File | Required Keys | Optional Keys |
|-----------|--------------|---------------|
| `test_llm.c` | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | - |
| `test_memory_llm.c` | `OPENAI_API_KEY` | - |

---

## Key Format Validation

### OpenAI
- **Format**: `sk-...`
- **Min Length**: 10 characters
- **Validation**: Checks for `sk-` prefix

### Anthropic
- **Format**: `sk-ant-...`
- **Min Length**: 10 characters
- **Validation**: Checks for `sk-ant-` prefix

### Azure OpenAI
- **Format**: Alphanumeric
- **Min Length**: 32 characters
- **Validation**: Length check only

### Google
- **Format**: No specific format
- **Min Length**: 10 characters
- **Validation**: Basic length check

---

## Where Keys Are Used

### Test Files
- `tests/test_llm.c` - Uses `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `tests/test_memory_llm.c` - Uses `OPENAI_API_KEY`

### Source Files
- `src/gv_llm.c` - LLM API calls (OpenAI, Anthropic, Azure, Custom)
- `src/gv_embedding.c` - Embedding API calls (OpenAI, Google, Custom)

### Python Bindings
- `python/src/gigavector/_core.py` - Python API wrapper

---

## Quick Start

### Minimum Setup (Tests Only)
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-key" > .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> .env

# Run tests
./build/test_llm
./build/test_memory_llm
```

### Full Setup (All Features)
```bash
# Create .env file with all keys
cat > .env << EOF
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-google-key
EOF
```

---

## Security Notes

1. **Never commit `.env` files** - Add to `.gitignore`
2. **Use environment variables** in production (not `.env` files)
3. **Rotate keys regularly** for security
4. **Use separate keys** for development and production
5. **Restrict API key permissions** when possible

---

## Additional Resources

- [OpenAI API Keys](https://platform.openai.com/api-keys)
- [Anthropic API Keys](https://console.anthropic.com/settings/keys)
- [Google AI Studio](https://aistudio.google.com)
- [Azure OpenAI Setup](https://learn.microsoft.com/azure/ai-services/openai/)

---

## Key Priority (How Keys Are Loaded)

1. **First**: Check `.env` file in current directory
2. **Second**: Check environment variables
3. **Third**: Use test/validation keys (format validation only)
4. **Last**: Skip test (if key required for real API call)

---

## Summary

### Required for Testing
- `OPENAI_API_KEY` - **REQUIRED**
- `ANTHROPIC_API_KEY` - **REQUIRED**

### Optional Features
- `GOOGLE_API_KEY` - For Google embeddings
- Azure OpenAI - Requires config (not env var)
- Custom providers - Require config (not env var)

### No Key Needed
- HuggingFace - Local/free models


