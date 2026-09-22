from observability.network_topology import (
    NetworkTopologyTracker,
    classify_network_location,
)


def test_classify_network_location_handles_local_and_public_values():
    assert classify_network_location("localhost") == "loopback"
    assert classify_network_location("127.0.0.1") == "loopback"
    assert classify_network_location("10.1.2.3") == "private-network"
    assert classify_network_location("8.8.8.8") == "public-internet"


def test_snapshot_groups_requests_by_registered_service(tmp_path):
    tracker = NetworkTopologyTracker(
        enabled=True,
        max_events=10,
        persist_path=str(tmp_path / "network-topology.jsonl"),
    )
    tracker.register_service(
        service="custom_stt",
        base_url="http://localhost:8080",
        provider="custom_stt",
    )

    with tracker.request_context(
        service="custom_stt",
        operation="transcribe",
        metadata={"language": "uz"},
    ):
        tracker.record_request(
            transport="aiohttp",
            method="POST",
            url="http://localhost:8080/transcribe/transcribe-batch",
            status_code=200,
            duration_ms=123.4,
        )

    snapshot = tracker.snapshot(limit=5)

    assert len(snapshot["services"]) == 1
    service = snapshot["services"][0]
    assert service["service"] == "custom_stt"
    assert service["stats"]["request_count"] == 1
    assert service["endpoint"]["host"] == "localhost"

    request = snapshot["recent_requests"][0]
    assert request["service"] == "custom_stt"
    assert request["operation"] == "transcribe"
    assert request["metadata"]["language"] == "uz"
