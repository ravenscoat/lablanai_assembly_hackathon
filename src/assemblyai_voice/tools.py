"""AssemblyAI tool schemas and safe RelayDesk dispatcher."""

from __future__ import annotations

from typing import Any

from relaydesk import RelayDeskStore
from relaydesk.orchestration import SPECIALISTS, specialist_context


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    properties = {
        **properties,
        "case_id": {
            "type": "string",
            "description": "Existing support case to receive this tool's evidence, when available.",
        },
    }
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


RELAYDESK_TOOLS = [
    _tool(
        "identify_customer",
        "Look up a RelayDesk customer by email before accessing account information.",
        {"email": {"type": "string"}},
        ["email"],
    ),
    _tool(
        "inspect_billing",
        "Read and verify captured charges, including duplicate purchase operations.",
        {"customer_id": {"type": "string"}},
        ["customer_id"],
    ),
    _tool(
        "refund_duplicate",
        "Refund exactly one verified duplicate charge. This operation is idempotent.",
        {
            "customer_id": {"type": "string"},
            "operation_id": {"type": "string"},
        },
        ["customer_id", "operation_id"],
    ),
    _tool(
        "inspect_permissions",
        "Read and verify a customer's membership and access to a project.",
        {
            "customer_id": {"type": "string"},
            "project_id": {"type": "string"},
        },
        ["customer_id", "project_id"],
    ),
    _tool(
        "restore_access",
        "Reactivate an existing project membership and verify the saved state.",
        {
            "customer_id": {"type": "string"},
            "project_id": {"type": "string"},
        },
        ["customer_id", "project_id"],
    ),
    _tool(
        "create_case",
        "Create durable shared case memory for a caller's issue.",
        {
            "customer_id": {"type": "string"},
            "issue": {"type": "string"},
        },
        ["issue"],
    ),
    _tool(
        "route_case",
        "Route a case while carrying confirmed facts to the next specialist.",
        {
            "case_id": {"type": "string"},
            "specialist": {
                "type": "string",
                "enum": ["front_desk", "billing", "subscriptions", "permissions"],
            },
            "confirmed_fact": {"type": "string"},
        },
        ["case_id", "specialist"],
    ),
    _tool(
        "get_case",
        "Read the full shared handoff packet for an existing support case.",
        {"case_id": {"type": "string"}},
        ["case_id"],
    ),
]


class RelayDeskToolDispatcher:
    def __init__(self, store: RelayDeskStore) -> None:
        self.store = store
        self._handlers = {
            "identify_customer": store.identify_customer,
            "inspect_billing": store.inspect_billing,
            "refund_duplicate": store.refund_duplicate,
            "inspect_permissions": store.inspect_permissions,
            "restore_access": store.restore_access,
            "create_case": store.create_case,
            "route_case": store.route_case,
            "get_case": store.get_case,
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        try:
            clean_arguments = dict(arguments)
            case_id = clean_arguments.get("case_id")
            if name in {
                "inspect_billing",
                "refund_duplicate",
                "inspect_permissions",
                "restore_access",
            }:
                if not case_id:
                    return {"ok": False, "error": "case_id is required for specialist tools"}
                packet = self.store.get_case(case_id)
                specialist = packet.get("current_specialist")
                profile = SPECIALISTS.get(specialist)
                if not profile or name not in profile.allowed_tools:
                    return {
                        "ok": False,
                        "error": f"Tool {name} is not allowed for specialist {specialist}",
                        "required_route": (
                            "billing" if "billing" in name or "refund" in name else "permissions"
                        ),
                    }
            if name not in {"get_case", "route_case"}:
                clean_arguments.pop("case_id", None)
            result = handler(**clean_arguments)
            if name == "route_case" and result.get("case_id"):
                result["specialist_context"] = specialist_context(
                    result["current_specialist"], result
                )
            if case_id and name not in {"create_case", "get_case", "route_case"}:
                self.store.record_case_event(case_id, name, result)
            return result
        except TypeError as exc:
            return {"ok": False, "error": f"Invalid arguments for {name}: {exc}"}
        except Exception as exc:
            return {"ok": False, "error": f"Tool {name} failed safely: {exc}"}
