"""
Custom LLM provider.
Uses local vLLM server with OpenAI-compatible API.
Endpoint: POST /v1/chat/completions
vLLM handles batching internally via continuous batching.
"""

from __future__ import annotations

import logging
import os

from livekit.plugins import openai

logger = logging.getLogger(__name__)


class CustomLLM(openai.LLM):
    """Custom LLM using a local vLLM server (OpenAI-compatible)."""

    def __init__(
        self,
        *,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0.2,
        max_completion_tokens: int = 2048,
    ) -> None:
        model = model or os.getenv("CUSTOM_LLM_MODEL", "gemma-4-E2B-it")
        api_key = api_key or os.getenv("CUSTOM_LLM_API_KEY", "not-needed")
        resolved_url = (base_url or os.getenv("CUSTOM_LLM_URL", "http://localhost:8003/v1")).rstrip(
            "/"
        )

        # Ensure the URL ends with /v1 for OpenAI compatibility
        if not resolved_url.endswith("/v1"):
            resolved_url = f"{resolved_url}/v1"

        logger.info(
            f"Custom LLM initialized: url={resolved_url}, " f"model={model}, temp={temperature}"
        )

        super().__init__(
            model=model,
            api_key=api_key or "not-needed",
            base_url=resolved_url,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
