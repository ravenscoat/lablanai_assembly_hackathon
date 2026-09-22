from pathlib import Path

from relaydesk import RelayDeskStore


def test_runtime_events_and_cases_feed_dashboard(tmp_path: Path) -> None:
    store = RelayDeskStore(tmp_path / "dashboard.db")
    case = store.create_case("cust_demo", "charged twice")
    store.record_runtime_event(
        "room-1", "tool_call", "inspect_billing", {"result": {"verified": True}}, case["case_id"]
    )
    assert store.list_runtime_events()[0]["payload"]["result"]["verified"] is True
    assert store.list_cases()[0]["case_id"] == case["case_id"]
