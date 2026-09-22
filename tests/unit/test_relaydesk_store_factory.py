from pathlib import Path

from relaydesk import RelayDeskStore, create_store


def test_store_factory_uses_sqlite_without_database_url(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "fallback.db"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("RELAYDESK_DB_PATH", str(path))
    store = create_store()
    assert isinstance(store, RelayDeskStore)
    assert store.identify_customer("alex@relaydesk.demo")["id"] == "cust_demo"
