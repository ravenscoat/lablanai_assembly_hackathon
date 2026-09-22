"""Schema validation for TransferConfig.out_of_hours_message."""

from __future__ import annotations


def test_transfer_config_accepts_out_of_hours_message():
    from config.schema import TransferConfig

    cfg = TransferConfig.model_validate(
        {
            "enabled": True,
            "out_of_hours_message": {
                "uz": "Ish vaqtidan tashqarida",
                "ru": "Вне рабочего времени",
            },
        }
    )
    assert cfg.out_of_hours_message["uz"] == "Ish vaqtidan tashqarida"
    assert cfg.out_of_hours_message["ru"] == "Вне рабочего времени"


def test_transfer_config_defaults_out_of_hours_message_to_empty_dict():
    from config.schema import TransferConfig

    cfg = TransferConfig.model_validate({"enabled": True})
    assert cfg.out_of_hours_message == {}
