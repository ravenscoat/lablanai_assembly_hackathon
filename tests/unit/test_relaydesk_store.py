from pathlib import Path

from assemblyai_voice.tools import RELAYDESK_TOOLS, RelayDeskToolDispatcher
from relaydesk import RelayDeskStore


def make_store(folder: Path) -> RelayDeskStore:
    return RelayDeskStore(folder / "relaydesk.db")


def test_detects_and_refunds_duplicate_exactly_once(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    before = store.inspect_billing("cust_demo")
    assert before["duplicate_operations"][0]["charge_count"] == 2

    first = store.refund_duplicate("cust_demo", "purchase_demo_100")
    second = store.refund_duplicate("cust_demo", "purchase_demo_100")

    assert first["verified"] is True
    assert first["captured_charges_after"] == 1
    assert second["idempotent_replay"] is True
    assert store.inspect_billing("cust_demo")["duplicate_operations"] == []


def test_restores_access_idempotently(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.inspect_permissions("cust_demo", "project_atlas")["active"] is False
    first = store.restore_access("cust_demo", "project_atlas")
    second = store.restore_access("cust_demo", "project_atlas")
    assert first["verified"] is True
    assert second["idempotent_replay"] is True
    assert store.inspect_permissions("cust_demo", "project_atlas")["active"] is True


def test_project_name_is_normalized_to_internal_identifier(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    result = store.inspect_permissions("cust_demo", "Project Atlas")
    assert result["verified"] is True
    assert result["project_id"] == "project_atlas"


def test_handoff_packet_carries_confirmed_context(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    case = store.create_case("cust_demo", "I was charged twice")
    packet = store.route_case(case["case_id"], "billing", "Identity confirmed by email")
    assert packet["current_specialist"] == "billing"
    assert "I was charged twice" in packet["confirmed_facts"]
    assert "Identity confirmed by email" in packet["confirmed_facts"]
    assert "do not ask" in packet["handoff_message"]


def test_tool_contract_is_flat_and_dispatcher_rejects_unknown_tool(tmp_path: Path) -> None:
    assert all(tool["type"] == "function" and "name" in tool for tool in RELAYDESK_TOOLS)
    assert all("function" not in tool for tool in RELAYDESK_TOOLS)
    dispatcher = RelayDeskToolDispatcher(make_store(tmp_path))
    assert dispatcher.execute("not_a_tool", {})["ok"] is False


def test_tool_evidence_is_written_to_shared_case_memory(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    dispatcher = RelayDeskToolDispatcher(store)
    case_id = store.create_case("cust_demo", "I was charged twice")["case_id"]
    store.route_case(case_id, "billing")
    result = dispatcher.execute("inspect_billing", {"customer_id": "cust_demo", "case_id": case_id})
    packet = store.get_case(case_id)
    assert result["verified"] is True
    assert "inspect_billing" in packet["completed_actions"]
    assert packet["evidence"][0]["action"] == "inspect_billing"


def test_specialist_tool_requires_correct_route_and_case(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    dispatcher = RelayDeskToolDispatcher(store)
    case_id = store.create_case("cust_demo", "I cannot open project Atlas")["case_id"]

    blocked = dispatcher.execute(
        "inspect_permissions", {"customer_id": "cust_demo", "project_id": "project_atlas"}
    )
    wrong_specialist = dispatcher.execute(
        "inspect_permissions",
        {"customer_id": "cust_demo", "project_id": "project_atlas", "case_id": case_id},
    )
    routed = dispatcher.execute("route_case", {"case_id": case_id, "specialist": "permissions"})
    allowed = dispatcher.execute(
        "inspect_permissions",
        {"customer_id": "cust_demo", "project_id": "project_atlas", "case_id": case_id},
    )

    assert blocked["ok"] is False and "case_id" in blocked["error"]
    assert wrong_specialist["ok"] is False
    assert "ACTIVE SPECIALIST: permissions" in routed["specialist_context"]
    assert allowed["verified"] is True
