"""
Flow document parser.
Parses .flow.md files with YAML frontmatter + markdown step sections.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from .models import FlowDocument, FlowOption, FlowStep, FlowValidation

logger = logging.getLogger(__name__)


class FlowParser:
    """Parses .flow.md files into FlowDocument objects."""

    @staticmethod
    def parse(path: Path) -> FlowDocument:
        """Parse a single .flow.md file."""
        content = path.read_text(encoding="utf-8")
        frontmatter, body = FlowParser._split_frontmatter(content)
        steps = FlowParser._parse_steps(body)

        return FlowDocument(
            id=frontmatter.get("id", path.stem),
            type=frontmatter.get("type", "guided"),
            title=frontmatter.get("title", ""),
            description=frontmatter.get("description", ""),
            trigger_keywords=frontmatter.get("trigger_keywords", []),
            activation_phrases=frontmatter.get("activation_phrases", []),
            faq_phrases=frontmatter.get("faq_phrases", []),
            initial_step=frontmatter.get("initial_step", ""),
            steps=steps,
            redirect_targets=frontmatter.get("redirect_targets", []),
            deterministic_response=frontmatter.get("deterministic_response"),
            deterministic_action=frontmatter.get("deterministic_action"),
            source_path=str(path),
        )

    @staticmethod
    def parse_all(flow_dir: Path) -> dict[str, FlowDocument]:
        """Parse all .flow.md files in a directory."""
        flows = {}
        if not flow_dir.exists():
            logger.warning(f"Flow directory not found: {flow_dir}")
            return flows

        for path in sorted(flow_dir.glob("*.flow.md")):
            try:
                flow = FlowParser.parse(path)
                flows[flow.id] = flow
                logger.debug(f"Parsed flow: {flow.id} ({flow.type}, " f"{len(flow.steps)} steps)")
            except Exception as e:
                logger.error(f"Failed to parse {path.name}: {e}")

        return flows

    @staticmethod
    def _split_frontmatter(content: str) -> tuple[dict, str]:
        """Split YAML frontmatter from markdown body."""
        pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)
        match = pattern.match(content)

        if not match:
            return {}, content

        frontmatter_str = match.group(1)
        body = match.group(2)

        try:
            frontmatter = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML frontmatter: {e}")
            frontmatter = {}

        return frontmatter, body

    @staticmethod
    def _parse_steps(body: str) -> dict[str, FlowStep]:
        """Parse ## step: sections from markdown body."""
        steps = {}

        # Split by ## step: headers
        step_pattern = re.compile(
            r"^## step:\s*(\w+)\s*$",
            re.MULTILINE,
        )

        parts = step_pattern.split(body)
        # parts = [pre-content, step_id_1, step_body_1, step_id_2, step_body_2, ...]

        for i in range(1, len(parts), 2):
            step_id = parts[i].strip()
            step_body = parts[i + 1] if i + 1 < len(parts) else ""

            try:
                step = FlowParser._parse_step_block(step_id, step_body)
                steps[step_id] = step
            except Exception as e:
                logger.error(f"Failed to parse step '{step_id}': {e}")

        return steps

    @staticmethod
    def _parse_step_block(step_id: str, body: str) -> FlowStep:
        """Parse a single step block's YAML-like content."""
        # Extract key-value pairs from the step body
        # The body is semi-structured: key: value lines + nested YAML blocks
        data = FlowParser._parse_step_yaml(body)

        # Parse options
        options = []
        if "options" in data and isinstance(data["options"], list):
            for opt in data["options"]:
                if isinstance(opt, dict):
                    options.append(
                        FlowOption(
                            label=opt.get("label", ""),
                            keywords=opt.get("keywords", []),
                            next_step=opt.get("next_step"),
                            redirect_flow=opt.get("redirect_flow"),
                            action=opt.get("action"),
                        )
                    )

        # Parse validation
        validation = None
        if "validation" in data and isinstance(data["validation"], dict):
            v = data["validation"]
            validation = FlowValidation(
                type=v.get("type", ""),
                min=v.get("min"),
                max=v.get("max"),
                allowed=v.get("allowed"),
                fail_message=v.get("fail_message", ""),
                fail_action=v.get("fail_action"),
            )

        # Determine step type
        step_type = data.get("type", "")
        if not step_type:
            if options:
                step_type = "choice"
            elif data.get("collect"):
                step_type = "collect"
            elif data.get("action") == "end_flow" or not data.get("next_step"):
                step_type = "terminal"
            else:
                step_type = "informational"

        return FlowStep(
            id=step_id,
            prompt=data.get("prompt", ""),
            type=step_type,
            options=options,
            collect=data.get("collect"),
            validation=validation,
            next_step=data.get("next_step"),
            action=data.get("action"),
            on_no_match=data.get("on_no_match"),
        )

    @staticmethod
    def _parse_step_yaml(body: str) -> dict:
        """
        Parse the semi-structured content of a step block.
        Handles both simple key: value and nested YAML blocks.
        """
        # Try to parse the entire body as YAML first
        try:
            # Clean up: remove markdown-only content (lines starting with #)
            lines = []
            for line in body.split("\n"):
                if line.strip().startswith("#") and not line.strip().startswith("# "):
                    continue
                lines.append(line)
            cleaned = "\n".join(lines).strip()

            if cleaned:
                result = yaml.safe_load(cleaned)
                if isinstance(result, dict):
                    return result
        except yaml.YAMLError:
            pass

        # Fallback: manual key-value extraction
        data = {}
        current_key = None
        current_value_lines = []

        for line in body.split("\n"):
            # Check for key: value pattern
            kv_match = re.match(r"^(\w+):\s*(.*)", line)
            if kv_match and not line.startswith("  "):
                # Save previous key
                if current_key:
                    data[current_key] = "\n".join(current_value_lines).strip()

                current_key = kv_match.group(1)
                value = kv_match.group(2).strip()

                if value.startswith("|"):
                    current_value_lines = []
                elif value:
                    data[current_key] = value
                    current_key = None
                    current_value_lines = []
                else:
                    current_value_lines = []
            elif current_key:
                current_value_lines.append(line)

        if current_key and current_value_lines:
            data[current_key] = "\n".join(current_value_lines).strip()

        return data
