from relaydesk.semantic_memory import case_memory_text


def test_case_memory_text_contains_operational_context() -> None:
    text = case_memory_text(
        {
            "confirmed_facts": ["charged twice"],
            "completed_actions": ["refund_duplicate"],
            "evidence": [{"verified": True}],
            "unresolved_tasks": [],
            "current_specialist": "billing",
        }
    )
    assert "charged twice" in text
    assert "refund_duplicate" in text
    assert "billing" in text
