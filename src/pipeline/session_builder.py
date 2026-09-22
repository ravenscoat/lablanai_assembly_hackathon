"""
Session builder: constructs AgentSession from config + pipeline components.
"""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import AgentSession

from config.schema import TenantConfig

logger = logging.getLogger(__name__)


class SessionBuilder:
    """Builds an AgentSession from TenantConfig and pipeline components."""

    @staticmethod
    def build(
        config: TenantConfig,
        *,
        stt: Any,
        tts: Any,
        llm: Any,
        vad: Any,
    ) -> AgentSession:
        """
        Build an AgentSession with all pipeline components wired.

        Args:
            config: Tenant configuration
            stt: Speech-to-text instance
            tts: Text-to-speech instance
            llm: Language model instance
            vad: Voice activity detection instance
        """
        logger.info(
            f"Building AgentSession for tenant={config.tenant.slug}: "
            f"stt={config.voice.stt_provider}, "
            f"tts={config.voice.tts_provider}, "
            f"llm={config.llm.provider}/{config.llm.model}"
        )

        return AgentSession(
            stt=stt,
            tts=tts,
            llm=llm,
            vad=vad,
            preemptive_generation=config.behavior.preemptive_generation,
        )
