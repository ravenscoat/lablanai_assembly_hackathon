from scripts.run_voice_evaluation import SCENARIOS


def test_evaluation_scenarios_cover_success_and_safety() -> None:
    assert {"duplicate_billing", "lost_access", "unsupported_request"} == set(SCENARIOS)
    assert "refund_duplicate" in SCENARIOS["duplicate_billing"]["required_tools"]
    assert "restore_access" in SCENARIOS["lost_access"]["required_tools"]
    assert SCENARIOS["unsupported_request"]["forbidden_tools"] == {
        "refund_duplicate",
        "restore_access",
    }
    assert SCENARIOS["unsupported_request"]["required_tools"] == {"identify_customer"}
