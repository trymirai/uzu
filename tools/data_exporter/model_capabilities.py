import json
from dataclasses import dataclass

from data_exporter.model import Model
from data_exporter.utils import BUNDLED_CONFIGS_PATH

__all__ = ["ModelCapabilities", "resolve_capabilities"]

BUNDLED_CONFIGS_MAPPING_PATH = BUNDLED_CONFIGS_PATH / "configs.json"


@dataclass(frozen=True)
class ContextKeyValue:
    key: str
    value: object


@dataclass(frozen=True)
class ReasoningCapabilities:
    enable: ContextKeyValue | None = None
    disable: ContextKeyValue | None = None

    @property
    def disableable(self) -> bool:
        return self.disable is not None


@dataclass(frozen=True)
class ToolsCapabilities:
    required: bool
    multiple: bool


@dataclass(frozen=True)
class ModelCapabilities:
    reasoning: ReasoningCapabilities | None = None
    tools: ToolsCapabilities | None = None
    required_system_prompt: str | None = None
    supports_system: bool = True

    @property
    def supports_reasoning(self) -> bool:
        return self.reasoning is not None

    @property
    def supports_tools(self) -> bool:
        return self.tools is not None


def load_bundled_configs_mapping() -> list[dict]:
    with open(BUNDLED_CONFIGS_MAPPING_PATH) as file:
        return json.load(file)


def load_rendering_config(rendering_name: str) -> dict:
    rendering_path = BUNDLED_CONFIGS_PATH / "rendering" / f"{rendering_name}.json"
    with open(rendering_path) as file:
        return json.load(file)


def load_ordering_config(ordering_name: str) -> dict:
    ordering_path = BUNDLED_CONFIGS_PATH / "ordering" / f"{ordering_name}.json"
    with open(ordering_path) as file:
        return json.load(file)


def is_role_avoidable(ordering: dict, role: str) -> bool:
    initial_roles = ordering.get("initial", [])
    transitions = ordering.get("transitions", {})
    has_other_initial_roles = any(initial_role != role for initial_role in initial_roles)
    all_transitions_have_alternatives = all(
        role not in target_roles or any(target_role != role for target_role in target_roles)
        for target_roles in transitions.values()
    )
    return has_other_initial_roles and all_transitions_have_alternatives


def resolve_hanashi_name(model: Model) -> str | None:
    for encoding in model.encodings:
        if encoding.get("type") == "hanashi":
            return encoding.get("name")
    return None


def find_role_and_field_by_block_name(
    rendering: dict,
    block_name: str,
) -> tuple[str, str, dict] | None:
    for role_key, role_config in rendering.items():
        if not isinstance(role_config, dict):
            continue
        for section_key in ("message", "context"):
            section = role_config.get(section_key, {})
            for field_name, field_config in section.items():
                if not isinstance(field_config, dict):
                    continue
                if field_config.get("block") == block_name:
                    return role_key, field_name, field_config
                if block_name in field_config.get("blocks", []):
                    return role_key, field_name, field_config
    return None


def find_field_by_block_name(rendering: dict, block_name: str) -> tuple[str, dict] | None:
    result = find_role_and_field_by_block_name(rendering, block_name)
    if result is None:
        return None
    _, field_name, field_config = result
    return field_name, field_config


def resolve_capabilities(model: Model) -> ModelCapabilities:
    hanashi_name = resolve_hanashi_name(model)
    if hanashi_name is None:
        return ModelCapabilities()

    bundled_mapping = load_bundled_configs_mapping()
    mapping = next(
        (entry for entry in bundled_mapping if entry["name"] == hanashi_name),
        None,
    )
    if mapping is None:
        return ModelCapabilities()

    rendering_config = load_rendering_config(mapping["rendering"])
    rendering = rendering_config.get("rendering", {})
    ordering_config = load_ordering_config(mapping["ordering"])

    reasoning = None
    if find_field_by_block_name(rendering, "reasoning") is not None:
        reasoning = ReasoningCapabilities()

        reasoning_effort_field = find_field_by_block_name(rendering, "reasoning_effort")
        if reasoning_effort_field is not None:
            field_name, field_config = reasoning_effort_field
            mapping_data = field_config.get("mapping", {})

            enable = None
            disable = None
            for mapping_key, mapping_value in mapping_data.items():
                if mapping_key == "disabled" and mapping_value is not None:
                    disable = ContextKeyValue(key=field_name, value=mapping_value)
                elif mapping_key == "default" and mapping_value is not None:
                    enable = ContextKeyValue(key=field_name, value=mapping_value)

            reasoning = ReasoningCapabilities(enable=enable, disable=disable)

    tools = None
    tools_role_and_field = find_role_and_field_by_block_name(rendering, "tools")
    if tools_role_and_field is not None:
        tools_role_key, _, tools_field_config = tools_role_and_field
        field_required = tools_field_config.get("required", False)
        role_avoidable = is_role_avoidable(ordering_config, tools_role_key)
        required = field_required and not role_avoidable

        multiple = True
        tool_calls_field = find_field_by_block_name(rendering, "tool_call")
        if tool_calls_field is not None:
            _, tool_calls_config = tool_calls_field
            limit = tool_calls_config.get("limit")
            if limit is not None and limit <= 1:
                multiple = False

        tools = ToolsCapabilities(required=required, multiple=multiple)

    supports_system = "system" in rendering

    return ModelCapabilities(
        reasoning=reasoning,
        tools=tools,
        supports_system=supports_system,
    )
