"""Qdrant point-id derivation must match the platform's kb-point-id.ts.

Both the agent's populate_qdrant.py and the platform's KB refresh worker write
to the same per-tenant Qdrant collection. A point representing the same FAQ
must get the SAME id from both writers, or a refresh prunes points the other
created (data loss). See NAV-231.
"""

from knowledge.point_id import KB_POINT_ID_NAMESPACE, point_id_for_external_id


def test_namespace_matches_platform():
    # MUST equal KB_POINT_ID_NAMESPACE in
    # navai-platform/services/api/src/services/kb-point-id.ts
    assert KB_POINT_ID_NAMESPACE == "a1d6e9f2-3b4c-5d6e-7f80-91a2b3c4d5e6"


def test_reference_vectors_match_platform():
    # Identical to the vectors pinned in kb-point-id.test.ts.
    assert point_id_for_external_id("d1") == "f2cf2176-fd12-5107-907f-61298aee0d1e"
    assert point_id_for_external_id("d2") == "8a598e75-0b4c-5a30-b3a2-b05b77b98ae3"
    assert point_id_for_external_id("hello") == "c8d802fd-fa08-579e-8678-dd3f57e34456"


def test_is_deterministic():
    assert point_id_for_external_id("faq_0") == point_id_for_external_id("faq_0")


def test_differs_per_external_id():
    assert point_id_for_external_id("faq_0") != point_id_for_external_id("faq_1")
