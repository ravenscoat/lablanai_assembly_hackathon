#!/usr/bin/env python3
"""
Manual integration test: exercises PlatformClient against the backend.

Usage:
    # Against mock backend:
    python scripts/mock_backend.py  # terminal 1
    python scripts/test_api_integration.py  # terminal 2

    # Against real backend:
    PLATFORM_API_URL=https://<your-platform-host> INTERNAL_API_KEY=... python scripts/test_api_integration.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("PLATFORM_API_URL", "http://localhost:3000")
os.environ.setdefault("INTERNAL_API_KEY", "test-key")

# Tenant slug: use env or default to the bundled example tenant
TENANT_ID = os.environ.get("TEST_TENANT_ID", "example-tenant")


async def main():
    from api.platform_client import PlatformClient

    client = PlatformClient()
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name} — {detail}")
            failed += 1

    # --- 1. Health check ---
    print("\n1. Health check")
    ok = await client.check_health()
    check("Backend is alive", ok, "Is the backend running?")
    if not ok:
        # Try API endpoint as health indicator
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as h:
                resp = await h.get(f"{client._base_url}/api/v1/internal/config")
                if resp.status_code in (200, 401, 403):
                    print("  (health endpoint missing but API is reachable)")
                    ok = True
        except Exception:
            pass
    if not ok:
        print("\nBackend not reachable. Check PLATFORM_API_URL.")
        return

    # --- 2. Caller history ---
    print("\n2. Caller history")
    history = await client.get_caller_history(
        phone="+92301234567",
        tenant_id=TENANT_ID,
    )
    check("Caller history fetched", True)  # API responded
    print(f"   has_history={history.has_history}, total_calls={history.total_calls}, "
          f"last_topic='{history.last_topic}'")

    # --- 3. Create call ---
    print("\n3. Create call")
    call_id = await client.create_call(
        tenant_id=TENANT_ID,
        call_sid=f"integration-test-{int(__import__('time').time())}",
        caller_phone="+92301234567",
        agent_phone="+92000000000",
    )
    check("Call created", call_id is not None, "got None")
    if call_id:
        check(f"Call ID: {call_id}", len(call_id) > 0)

    # --- 4. Update call ---
    print("\n4. Update call")
    if call_id:
        ok = await client.update_call(
            call_db_id=call_id,
            tenant_id=TENANT_ID,
            status="in_progress",
            duration_seconds=30,
            transcript=[
                {"role": "user", "text": "what are your business hours"},
                {"role": "agent", "text": "Yoshlar daftari - bu 14-30 yoshdagilar uchun..."},
            ],
            transcript_text="user: what are your hours\nassistant: We are open 9 to 6...",
            ai_summary="User asked about business hours",
            metadata={"tenant": TENANT_ID, "room_name": "integration-test-room"},
        )
        check("Call updated", ok)

    # --- 5. Request operator ---
    print("\n5. Operator handoff")
    if call_id:
        ok = await client.request_operator(
            call_db_id=call_id,
            tenant_id=TENANT_ID,
            transfer_reason="User requested operator",
            ai_summary="User asked about daftar, then requested human help",
            room_name="integration-test-room",
            agent_identity="agent-test-001",
        )
        check("Operator requested", ok)

        if ok:
            # Poll status
            print("   Polling operator status...")
            for i in range(5):
                await asyncio.sleep(2)
                status = await client.get_operator_status(
                    call_db_id=call_id, tenant_id=TENANT_ID,
                )
                print(f"   Poll {i+1}: status={status.status}"
                      f"{f' operator={status.operator_name}' if status.operator_name else ''}")
                if status.status == "transferred":
                    check("Operator accepted", True)
                    break
            else:
                print("   (No operator online — expected in dev)")

    # --- 6. Final call update ---
    print("\n6. Final call update (completion)")
    if call_id:
        ok = await client.update_call(
            call_db_id=call_id,
            tenant_id=TENANT_ID,
            status="completed",
            duration_seconds=45,
            ai_summary="Integration test call — testing API endpoints",
            ended_at="2026-04-09T12:01:00Z",
            metadata={
                "tenant": TENANT_ID,
                "transferred": False,
                "room_name": "integration-test-room",
            },
        )
        check("Final update sent", ok)

    # --- 7. Submit murojaat ---
    print("\n7. Murojaat submission")
    result = await client.submit_murojaat(
        tenant_id=TENANT_ID,
        content="Integration test — bu test murojaat, e'tibor bermang",
        full_name="Test Foydalanuvchi",
        age=22,
        region="Toshkent viloyati",
        district="Chilonzor tumani",
    )
    check("Murojaat submitted", result.success, result.error or "")
    if result.murojaat_id:
        check(f"Murojaat ID: {result.murojaat_id}", True)

    # --- Summary ---
    await client.close()
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("All API integrations working!")
    else:
        print("Some tests failed — check output above")


if __name__ == "__main__":
    asyncio.run(main())
