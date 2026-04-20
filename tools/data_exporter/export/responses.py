import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import product

from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging
from transformers.utils import chat_template_utils
from transformers.utils.chat_template_utils import get_json_schema

from data_exporter.export.bundled import process_chat_template_model_specific, resolve_chat_template
from data_exporter.model_capabilities import (
    ModelCapabilities,
    resolve_capabilities,
)
from data_exporter.utils import CONFIGS_PATH, RESPONSES_PATH, load_registry

__all__ = ["export_responses"]

_original_get_json_schema_type = chat_template_utils._get_json_schema_type  # noqa: SLF001


def _patched_get_json_schema_type(param_type: type) -> dict[str, str]:
    if isinstance(param_type, type) and hasattr(param_type, "json_schema"):
        return param_type.json_schema()
    return _original_get_json_schema_type(param_type)


chat_template_utils._get_json_schema_type = _patched_get_json_schema_type  # noqa: SLF001


@dataclass(frozen=True)
class Location:
    city: str
    country: str

    @staticmethod
    def json_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
                "country": {"type": "string", "description": "The country name"},
            },
            "required": ["city", "country"],
        }


def get_current_temperature(location: Location, unit: str) -> float:  # noqa: ARG001
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    return 30.0


def get_current_wind_speed(location: str) -> float:  # noqa: ARG001
    """
    Get the current wind speed in km/h at a given location.

    Args:
        location: The location to get the wind speed for, in the format "City, Country"

    Returns:
        The current wind speed in km/h
    """
    return 10.0


@dataclass(frozen=True)
class TestMessages:
    messages: list[dict]
    required_tools: list[Callable]


TEST_MESSAGES_LIST: list[TestMessages] = [
    TestMessages(
        messages=[
            {"role": "user", "content": "What is the current temperature in London?"},
        ],
        required_tools=[get_current_temperature],
    ),
    TestMessages(
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the current wind speed in London?"},
        ],
        required_tools=[get_current_wind_speed],
    ),
    TestMessages(
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the current temperature and wind speed in London?"},
        ],
        required_tools=[get_current_temperature, get_current_wind_speed],
    ),
]


@dataclass(frozen=True)
class ResponseTestParameters:
    messages: list[dict]
    tools: list[dict] | None
    context: dict


@dataclass(frozen=True)
class ResponseTestResult:
    prompt: str
    completion: str


@dataclass(frozen=True)
class ResponseTestExpectations:
    reasoning: bool
    tool_call_names: list[str]


@dataclass(frozen=True)
class ResponseTestData:
    repo_id: str
    parameters: ResponseTestParameters
    result: ResponseTestResult
    expectations: ResponseTestExpectations


type Variation = tuple[TestMessages, list[Callable] | None, dict]


def apply_required_system_prompt(
    test_messages_list: list[TestMessages],
    required_system_prompt: str,
) -> list[TestMessages]:
    result = []
    for test_messages in test_messages_list:
        messages = [message for message in test_messages.messages if message.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": required_system_prompt})
        result.append(
            TestMessages(
                messages=messages,
                required_tools=test_messages.required_tools,
            ),
        )
    return result


TRANSLATION_TEST_MESSAGES_LIST: list[TestMessages] = [
    TestMessages(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "fr",
                        "text": "Tell me about London.",
                    },
                ],
            },
        ],
        required_tools=[],
    ),
    TestMessages(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "fr",
                        "target_lang_code": "en",
                        "text": "Parlez-moi de Londres.",
                    },
                ],
            },
        ],
        required_tools=[],
    ),
    TestMessages(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "de-DE",
                        "text": "The weather in Paris is lovely today.",
                    },
                ],
            },
        ],
        required_tools=[],
    ),
    TestMessages(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "de",
                        "target_lang_code": "en",
                        "text": "Das Wetter in Paris ist heute wunderschön.",
                    },
                ],
            },
        ],
        required_tools=[],
    ),
]


def build_variations(capabilities: ModelCapabilities, is_translation: bool = False) -> list[Variation]:
    # In a perfect world, we would filter messages by multiple tool call support,
    # but for testing purposes we want to cover more edge cases.
    messages_list = TRANSLATION_TEST_MESSAGES_LIST if is_translation else TEST_MESSAGES_LIST
    if not capabilities.supports_system:
        messages_list = [
            test_messages
            for test_messages in messages_list
            if not any(message.get("role") == "system" for message in test_messages.messages)
        ]
    if capabilities.required_system_prompt is not None:
        messages_list = apply_required_system_prompt(
            messages_list,
            capabilities.required_system_prompt,
        )

    tools_variations: list[list[Callable] | None] = []
    if capabilities.tools is not None:
        if not capabilities.tools.required:
            tools_variations.append(None)
            tools_variations.append([])
        tools_variations.append([get_current_temperature])
        tools_variations.append([get_current_wind_speed])
        tools_variations.append([get_current_temperature, get_current_wind_speed])
    else:
        tools_variations.append(None)

    context_variations: list[dict] = []
    if capabilities.reasoning is not None:
        if capabilities.reasoning.enable is not None:
            context_variations.append(
                {capabilities.reasoning.enable.key: capabilities.reasoning.enable.value},
            )
        if capabilities.reasoning.disable is not None:
            context_variations.append(
                {capabilities.reasoning.disable.key: capabilities.reasoning.disable.value},
            )
    if not context_variations:
        context_variations.append({})

    return list(product(messages_list, tools_variations, context_variations))


def expect_reasoning(capabilities: ModelCapabilities, context: dict) -> bool:
    if capabilities.reasoning is None:
        return False
    return not (
        capabilities.reasoning.disable is not None
        and context.get(capabilities.reasoning.disable.key) == capabilities.reasoning.disable.value
    )


def expect_tool_call_names(
    test_messages: TestMessages,
    tools: list[Callable] | None,
) -> list[str]:
    if tools is None or len(tools) == 0:
        return []
    return [tool.__name__ for tool in test_messages.required_tools if tool in tools]


def export_responses(console: Console, err_console: Console) -> None:
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    RESPONSES_PATH.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    for model in registry:
        if model.repo_id in ["mistralai/Mistral-Small-3.1-24B-Instruct-2503", "mistralai/Devstral-Small-2505"]:
            continue
        try:
            capabilities = resolve_capabilities(model)
            required_system_prompt = None
            if model.repo_id == "google/functiongemma-270m-it":
                required_system_prompt = "You are a model that can do function calling with the following functions"
            if model.repo_id == "mistralai/Devstral-Small-2505":
                required_system_prompt = "You are Devstral, a helpful agentic model trained by Mistral AI and using the OpenHands scaffold. You can interact with a computer to solve tasks.\n\n<ROLE>\nYour primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.\n* If the user asks a question, like \"why is X happening\", don't try to fix the problem. Just give an answer to the question.\n</ROLE>\n\n<EFFICIENCY>\n* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.\n* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.\n</EFFICIENCY>\n\n<FILE_SYSTEM_GUIDELINES>\n* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.\n* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.\n* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.\n</FILE_SYSTEM_GUIDELINES>\n\n<CODE_QUALITY>\n* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.\n* When implementing solutions, focus on making the minimal changes needed to solve the problem.\n* Before implementing any changes, first thoroughly understand the codebase through exploration.\n* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.\n</CODE_QUALITY>\n\n<VERSION_CONTROL>\n* When configuring git credentials, use \"openhands\" as the user.name and \"openhands@all-hands.dev\" as the user.email by default, unless explicitly instructed otherwise.\n* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.\n* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.\n* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.\n* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.\n</VERSION_CONTROL>\n\n<PULL_REQUESTS>\n* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.\n* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.\n* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.\n</PULL_REQUESTS>\n\n<PROBLEM_SOLVING_WORKFLOW>\n1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions\n2. ANALYSIS: Consider multiple approaches and select the most promising one\n3. TESTING:\n   * For bug fixes: Create tests to verify issues before implementing fixes\n   * For new features: Consider test-driven development when appropriate\n   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure\n   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies\n4. IMPLEMENTATION: Make focused, minimal changes to address the problem\n5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.\n</PROBLEM_SOLVING_WORKFLOW>\n\n<SECURITY>\n* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.\n* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.\n</SECURITY>\n\n<ENVIRONMENT_SETUP>\n* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.\n* If you encounter missing dependencies:\n  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)\n  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)\n  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed\n* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.\n</ENVIRONMENT_SETUP>\n\n<TROUBLESHOOTING>\n* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:\n  1. Step back and reflect on 5-7 different possible sources of the problem\n  2. Assess the likelihood of each possible cause\n  3. Methodically address the most likely causes, starting with the highest probability\n  4. Document your reasoning process\n* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.\n</TROUBLESHOOTING>"
            if required_system_prompt is not None:
                capabilities = ModelCapabilities(
                    reasoning=capabilities.reasoning,
                    tools=capabilities.tools,
                    required_system_prompt=required_system_prompt,
                )
            variations = build_variations(capabilities, is_translation=model.is_translation)

            test_data_path = RESPONSES_PATH / f"{model.name}.json"
            if test_data_path.exists():
                with open(test_data_path) as file:
                    try:
                        existing_test_data = json.load(file)
                        if len(existing_test_data) == len(variations):
                            console.print(f"[green]{model.repo_id} completed[/green]")
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                test_data_path.unlink()

            test_data_list = []

            for test_messages, tools, context in track(
                variations,
                description=model.repo_id,
                console=console,
            ):
                tool_schemas = [get_json_schema(tool) for tool in tools] if tools is not None else None

                model_path = CONFIGS_PATH / model.tokenizer_name
                with open(model_path / "tokenizer_config.json") as file:
                    tokenizer_config = json.load(file)
                chat_template = resolve_chat_template(model_path, tokenizer_config)
                assert chat_template is not None
                chat_template = process_chat_template_model_specific(
                    chat_template,
                    model.repo_id,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    model.tokenizer_repo_id if model.tokenizer_repo_id is not None else model.repo_id,
                )
                tokenizer.chat_template = process_chat_template_model_specific(
                    chat_template,
                    model.repo_id,
                )
                prompt = tokenizer.apply_chat_template(
                    test_messages.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tool_schemas,
                    **context,
                )
                tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")

                chat_model = AutoModelForCausalLM.from_pretrained(model.repo_id)
                outputs = chat_model.generate(tokenized_prompt, max_new_tokens=2048)
                completion = tokenizer.decode(outputs[0][tokenized_prompt.shape[-1] :])

                test_data = ResponseTestData(
                    repo_id=model.repo_id,
                    parameters=ResponseTestParameters(
                        messages=test_messages.messages,
                        tools=tool_schemas,
                        context=context,
                    ),
                    result=ResponseTestResult(
                        prompt=prompt,
                        completion=completion,
                    ),
                    expectations=ResponseTestExpectations(
                        reasoning=expect_reasoning(capabilities, context),
                        tool_call_names=expect_tool_call_names(test_messages, tools),
                    ),
                )
                test_data_list.append(asdict(test_data))

            with open(test_data_path, "w") as file:
                json.dump(test_data_list, file, indent=4)

            console.print(f"[green]{model.repo_id} completed[/green]")
        except Exception as error:  # noqa: BLE001
            err_console.print(f"[red]{model.repo_id} failed[/red]: {error}")
