from __future__ import annotations

from config.loader import ConfigLoader


def test_normalize_phone_supports_sip_uri():
    normalized = ConfigLoader._normalize_phone("sip:+923001234567@pbx.local;user=phone")
    assert normalized == "+923001234567"


def test_normalize_phone_supports_tel_uri():
    normalized = ConfigLoader._normalize_phone("tel:+92-300-123-4567")
    assert normalized == "+923001234567"


def test_normalize_phone_preserves_extension_prefix():
    normalized = ConfigLoader._normalize_phone("ext: 701")
    assert normalized == "ext:701"
