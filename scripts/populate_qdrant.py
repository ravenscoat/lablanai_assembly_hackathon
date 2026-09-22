#!/usr/bin/env python3
"""
Populate Qdrant with FAQ data for tenant knowledge bases.

Reads FAQ entries from JSON source file, embeds with Gemini (free tier),
and uploads to a Qdrant collection per tenant.

Usage:
    # Populate a tenant KB from its FAQ source file
    python scripts/populate_qdrant.py \
        --source knowledge/example-tenant/faq_source.json \
        --collection example_tenant_faq

    # Specify collection name
    python scripts/populate_qdrant.py --source faq.json --collection my_tenant_faq

    # Force recreate collection
    python scripts/populate_qdrant.py --source faq.json --force

Required env vars:
    QDRANT_URL=http://localhost:6333

Auth: uses Vertex AI via Application Default Credentials. On the VM this is
the attached service account. Locally, run `gcloud auth application-default login`.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from knowledge.point_id import point_id_for_external_id

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # Matryoshka — 768 keeps ~99% retrieval quality at 4x smaller than default 3072


def load_faqs_from_json(filepath: str) -> list[dict]:
    """Load FAQs from JSON file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    faqs = []
    for item in data:
        faq = {
            "id": item.get("id", f"faq_{len(faqs)}"),
            "category": item.get("category", "general"),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "keywords": item.get("keywords", []),
            "source": item.get("source", "unknown"),
        }
        # Searchable text = question + keywords + answer
        parts = [faq["question"], " ".join(faq["keywords"]), faq["answer"]]
        faq["text"] = " ".join(p for p in parts if p)
        faqs.append(faq)

    logger.info(f"Loaded {len(faqs)} FAQs from {filepath}")
    return faqs


async def embed_texts(texts: list[str], batch_size: int = 10) -> list[list[float]]:
    """Embed texts using Gemini. Prefers Google AI Studio (GOOGLE_API_KEY) when set,
    otherwise falls back to Vertex AI ADC for the VM-attached service account.
    """
    try:
        from google import genai
        from google.genai.types import EmbedContentConfig
    except ImportError:
        raise ImportError("google-genai package required. Install: pip install google-genai")

    if GOOGLE_API_KEY:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("Using google.genai SDK (AI Studio API key) for embeddings")
    else:
        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
        logger.info("Using google.genai SDK (Vertex) for embeddings")

    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    embed_config = EmbedContentConfig(output_dimensionality=EMBEDDING_DIM)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

        for text in batch:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=embed_config,
            )
            embeddings.append(result.embeddings[0].values)

        # Rate limit: Gemini free tier allows ~1500 RPM
        if batch_num < total_batches:
            await asyncio.sleep(0.5)

    logger.info(f"Embedded {len(embeddings)} texts (dim={len(embeddings[0])})")
    return embeddings


async def upload_to_qdrant(
    faqs: list[dict],
    embeddings: list[list[float]],
    collection: str,
    force: bool = False,
):
    """Upload FAQs with embeddings to Qdrant."""
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = AsyncQdrantClient(url=QDRANT_URL)

    # Check/create collection
    collections = await client.get_collections()
    existing = [c.name for c in collections.collections]

    if collection in existing:
        if force:
            await client.delete_collection(collection)
            logger.info(f"Deleted existing collection '{collection}'")
        else:
            logger.info(f"Collection '{collection}' exists, upserting")

    if collection not in existing or force:
        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=len(embeddings[0]),
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection '{collection}' (dim={len(embeddings[0])})")

    # Build points.
    # Qdrant point id is the deterministic UUIDv5 of the FAQ's stable id, the
    # same scheme the platform's KB refresh worker uses (kb-point-id.ts), so a
    # later refresh upserts/prunes by matching ids instead of wiping this
    # collection. The external id stays in the payload as `id`. See NAV-231.
    points = [
        PointStruct(
            id=point_id_for_external_id(str(faq["id"])),
            vector=emb,
            payload={
                "id": faq["id"],
                "category": faq["category"],
                "question": faq["question"],
                "text": faq["answer"],  # 'text' is what RAG returns
                "answer": faq["answer"],
                "keywords": faq["keywords"],
                "source": faq.get("source", "unknown"),
            },
        )
        for faq, emb in zip(faqs, embeddings)
    ]

    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await client.upsert(collection_name=collection, points=batch)

    info = await client.get_collection(collection)
    logger.info(f"Uploaded {len(points)} points to '{collection}' (total: {info.points_count})")


async def _already_populated(collection: str, expected_count: int) -> bool:
    """True if `collection` exists in Qdrant and has exactly `expected_count` points.

    Used by --auto to skip embedding when the collection already mirrors the
    source JSON. Returns False on any error (missing collection, network) so
    callers fall through to repopulate.
    """
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=QDRANT_URL)
    try:
        info = await client.get_collection(collection)
        return info.points_count == expected_count
    except Exception:
        return False


async def main():
    parser = argparse.ArgumentParser(description="Populate Qdrant with FAQ data")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str, help="JSON file with FAQs")
    group.add_argument(
        "--auto", action="store_true", help="Auto-repopulate tenants from YAML config"
    )

    parser.add_argument("--collection", default="example_tenant_faq", help="Qdrant collection name")
    parser.add_argument("--force", action="store_true", help="Force recreate collection")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("KB Populator (Gemini embeddings)")
    logger.info(f"  Qdrant:     {QDRANT_URL}")
    if not args.auto:
        logger.info(f"  Collection: {args.collection}")
        logger.info(f"  Source:     {args.source}")
    else:
        logger.info("  Mode:       Auto (YAML Config)")
    logger.info("=" * 50)

    if args.auto:
        defaults_path = Path(__file__).parent.parent / "configs" / "tenants" / "_defaults.yaml"
        if not defaults_path.exists():
            logger.error(f"Cannot find {defaults_path}")
            sys.exit(1)

        with open(defaults_path, encoding="utf-8") as f:
            defaults = yaml.safe_load(f)

        repopulate = defaults.get("repopulate_kb", {})

        for kb_dir, enabled in repopulate.items():
            if not enabled:
                continue

            collection_name = f"{kb_dir}_faq"
            src_path = (
                Path(__file__).parent.parent
                / "knowledge"
                / kb_dir.replace("_", "-")
                / "faq_source.json"
            )
            if not src_path.exists():
                logger.warning(f"Source file {src_path} does not exist, skipping {kb_dir}")
                continue

            faqs = load_faqs_from_json(str(src_path))
            if not faqs:
                logger.warning(f"No FAQs in {src_path}, skipping {kb_dir}")
                continue

            # Idempotency: skip embed+upload if the collection already has the
            # expected number of points. Saves ~30-60s of Vertex embedding
            # cost per container restart. Pass --force to override.
            if not args.force and await _already_populated(collection_name, len(faqs)):
                logger.info(
                    f"[{kb_dir}] collection '{collection_name}' already has "
                    f"{len(faqs)} points — skipping (pass --force to repopulate)"
                )
                continue

            logger.info(f"Auto-repopulating KB for {kb_dir} ({len(faqs)} FAQs)")
            texts = [faq["text"] for faq in faqs]
            embeddings = await embed_texts(texts)
            await upload_to_qdrant(faqs, embeddings, collection_name, force=args.force)

    else:
        faqs = load_faqs_from_json(args.source)
        if not faqs:
            logger.error("No FAQs found!")
            sys.exit(1)

        texts = [faq["text"] for faq in faqs]
        embeddings = await embed_texts(texts)
        await upload_to_qdrant(faqs, embeddings, args.collection, args.force)

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
