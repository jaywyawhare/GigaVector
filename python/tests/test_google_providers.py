"""Tests for Google/Gemini provider wiring in Python bindings."""

from __future__ import annotations

import os
import unittest

from gigavector import (
    AgentConfig,
    AutoEmbedConfig,
    AutoEmbedProvider,
    EmbeddingConfig,
    EmbeddingProvider,
    InferenceConfig,
    LLMConfig,
    LLMProvider,
)


class TestGoogleProviders(unittest.TestCase):
    def test_embedding_provider_google_enum(self) -> None:
        self.assertEqual(EmbeddingProvider.GOOGLE, 4)
        self.assertEqual(int(EmbeddingProvider.GOOGLE), 4)

    def test_auto_embed_provider_google_enum(self) -> None:
        self.assertEqual(AutoEmbedProvider.GOOGLE, 1)
        self.assertNotEqual(AutoEmbedProvider.GOOGLE, EmbeddingProvider.GOOGLE)

    def test_embedding_config_google_provider_value(self) -> None:
        config = EmbeddingConfig(
            provider=EmbeddingProvider.GOOGLE,
            api_key="test-key",
            model="text-embedding-004",
            embedding_dimension=768,
        )
        c_config, _refs = config._to_c_config()
        self.assertEqual(int(c_config.provider), 4)

    def test_agent_config_llm_provider_is_string(self) -> None:
        config = AgentConfig(llm_provider="google", api_key="test-key")
        self.assertEqual(config.llm_provider, "google")

    def test_inference_config_embed_provider_is_string(self) -> None:
        config = InferenceConfig(embed_provider="google", api_key="test-key")
        self.assertEqual(config.embed_provider, "google")
        self.assertEqual(config.dimension, 0)

    def test_auto_embed_google_defaults_applied(self) -> None:
        from gigavector._core import _auto_embed_model_and_dimension

        model, dim = _auto_embed_model_and_dimension(
            AutoEmbedProvider.GOOGLE, "text-embedding-3-small", 1536
        )
        self.assertEqual(model, "text-embedding-004")
        self.assertEqual(dim, 768)

    def test_llm_config_google_provider(self) -> None:
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            api_key="test-google-api-key-1234567890",
            model="gemini-2.5-flash",
        )
        c_config, _refs = config._to_c_config()
        self.assertEqual(int(c_config.provider), 2)

    @unittest.skipUnless(
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
        "GOOGLE_API_KEY not set",
    )
    def test_auto_embedder_google_live(self) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        from gigavector import AutoEmbedder

        embedder = AutoEmbedder(
            AutoEmbedConfig(
                provider=AutoEmbedProvider.GOOGLE,
                api_key=api_key,
                model_name="text-embedding-004",
                dimension=768,
            )
        )
        try:
            vec = embedder.embed_text("hello world")
            self.assertEqual(len(vec), 768)
        finally:
            embedder.close()


if __name__ == "__main__":
    unittest.main()
