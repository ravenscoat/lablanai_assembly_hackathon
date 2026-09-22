from pathlib import Path

from relaydesk import RelayDeskStore
from relaydesk.dashboard import HTML, EvaluationController


def test_runtime_events_and_cases_feed_dashboard(tmp_path: Path) -> None:
    store = RelayDeskStore(tmp_path / "dashboard.db")
    case = store.create_case("cust_demo", "charged twice")
    store.record_runtime_event(
        "room-1", "tool_call", "inspect_billing", {"result": {"verified": True}}, case["case_id"]
    )
    assert store.list_runtime_events()[0]["payload"]["result"]["verified"] is True
    assert store.list_cases()[0]["case_id"] == case["case_id"]


def test_dashboard_is_an_interactive_voice_product(tmp_path: Path) -> None:
    assert "Start real voice call" in HTML
    assert "Live call playground" in HTML
    assert "Agent orchestration" in HTML
    assert "Persistent case memory" in HTML
    assert "Tool execution" in HTML
    assert "Independent verifier" in HTML
    assert "audio frames" in HTML
    controller = EvaluationController(tmp_path)
    started, response = controller.start("not-a-real-scenario")
    assert started is False
    assert response["error"] == "Unknown evaluation scenario"
