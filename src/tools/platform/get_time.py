"""
get_current_time tool: returns the current date/time in the tenant's office
timezone (transfer.timezone, default Asia/Karachi).

TODO(urdu): the day/month names and the spoken sentence below are in Uzbek.
Add an 'ur' (and 'en') localized variant when wiring an Urdu tenant — this is
one of the localized framework fragments noted in the README "Urdu integration
seams" section.
"""

from __future__ import annotations

from livekit.agents import RunContext, function_tool

from config.schema import TenantConfig
from tools.platform.escalate import DEFAULT_TZ


def create_time_tool(config: TenantConfig, **kwargs):
    """Factory: creates a get_current_time function_tool."""
    tz_name = (getattr(config.transfer, "timezone", None) or DEFAULT_TZ).strip() or DEFAULT_TZ

    @function_tool(name="get_current_time")
    async def get_current_time(context: RunContext) -> str:
        """Hozirgi vaqtni ayting. Get the current date and time."""
        from datetime import datetime

        import pytz

        tz = pytz.timezone(tz_name)
        now = datetime.now(tz)

        weekdays = [
            "Dushanba",
            "Seshanba",
            "Chorshanba",
            "Payshanba",
            "Juma",
            "Shanba",
            "Yakshanba",
        ]
        months = [
            "",
            "yanvar",
            "fevral",
            "mart",
            "aprel",
            "may",
            "iyun",
            "iyul",
            "avgust",
            "sentyabr",
            "oktyabr",
            "noyabr",
            "dekabr",
        ]

        day_name = weekdays[now.weekday()]
        month_name = months[now.month]
        time_str = now.strftime("%H:%M")

        return (
            f"Bugun {day_name}, {now.day}-{month_name} {now.year}-yil. "
            f"Hozir soat {time_str} (Toshkent vaqti)."
        )

    return get_current_time
