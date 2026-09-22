"""Pick which Vertex AI region to talk to for this project.

Tries `europe-west4` first (better latency for our EU traffic). If the
configured model isn't published there for the project, falls back to
`global`. Operators can short-circuit the probe by setting
`GOOGLE_CLOUD_LOCATION` — used by scripts and the eval pipeline too.

The probe runs once per (model) and is cached process-wide; production
boot pays a single `models.get` round-trip the first time each model
is needed.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

# GCP project for Vertex AI (Gemini reasoning + embeddings). Env-driven — set
# GOOGLE_CLOUD_PROJECT to your project id. Empty means "let the google client
# resolve it from ADC / the attached service account".
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
PRIMARY_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION_PRIMARY", "europe-west4")
FALLBACK_LOCATION = "global"


@lru_cache(maxsize=8)
def resolve_location(model: str) -> str:
    override = os.getenv("GOOGLE_CLOUD_LOCATION")
    if override:
        logger.info(f"Vertex location: {override} (from GOOGLE_CLOUD_LOCATION) for {model}")
        return override

    from google import genai

    try:
        client = genai.Client(vertexai=True, project=PROJECT, location=PRIMARY_LOCATION)
        client.models.get(model=model)
        logger.info(f"Vertex location: {PRIMARY_LOCATION} (primary) for {model}")
        return PRIMARY_LOCATION
    except Exception as exc:
        logger.warning(
            f"Vertex {PRIMARY_LOCATION} unavailable for {model} ({exc}); "
            f"falling back to {FALLBACK_LOCATION}"
        )
        return FALLBACK_LOCATION


def vertex_endpoint(location: str) -> str:
    """Aiplatform URL for the given location (used by network observability)."""
    if location == "global":
        return "https://aiplatform.googleapis.com"
    return f"https://{location}-aiplatform.googleapis.com"
