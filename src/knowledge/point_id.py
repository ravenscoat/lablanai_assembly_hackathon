"""Deterministic Qdrant point ids derived from a KB entry's external id.

Qdrant only accepts point ids that are an unsigned 64-bit integer or a UUID.
A FAQ's stable identifier (``faq["id"]`` in faq_source.json, e.g. ``faq_3``) is
neither, so we map it to a deterministic UUIDv5.

The namespace MUST stay identical to ``KB_POINT_ID_NAMESPACE`` in
``navai-platform/services/api/src/services/kb-point-id.ts``. Both the agent's
populate_qdrant.py and the platform's KB refresh worker write to the same
per-tenant collection; if they derive different ids for the same FAQ, a refresh
prunes points the other writer created (data loss). See NAV-231.
"""

import uuid

KB_POINT_ID_NAMESPACE = "a1d6e9f2-3b4c-5d6e-7f80-91a2b3c4d5e6"

_NAMESPACE = uuid.UUID(KB_POINT_ID_NAMESPACE)


def point_id_for_external_id(external_id: str) -> str:
    """Return the UUIDv5 of ``external_id`` under :data:`KB_POINT_ID_NAMESPACE`."""
    return str(uuid.uuid5(_NAMESPACE, external_id))
