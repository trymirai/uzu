import asyncio
import json

from pydantic import BaseModel

from uzu import (
    ChatConfig,
    ChatMessage,
    ChatReplyConfig,
    Engine,
    EngineConfig,
    Grammar,
    ReasoningEffort,
)


class Country(BaseModel):
    name: str
    capital: str


class CountryList(BaseModel):
    countries: list[Country]


def structured_response(response: str | None, model_type: type[BaseModel]) -> BaseModel | None:
    if not response:
        return None
    return model_type.model_validate_json(response)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("Qwen/Qwen3-0.6B")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"Download progress: {update.progress()}")

    schema_string = json.dumps(CountryList.model_json_schema())
    messages = [
        ChatMessage.system().with_reasoning_effort(ReasoningEffort.Disabled),
        ChatMessage.user().with_text(
            "Give me a JSON object containing a list of 3 countries, where each country has name and capital fields"
        ),
    ]

    session = await engine.chat(model, ChatConfig.create())
    replies = await session.reply(
        messages,
        ChatReplyConfig.create().with_grammar(Grammar.JsonSchema(schema_string)),
    )
    if replies:
        countries = structured_response(replies[0].message.text, CountryList)
        print(countries)


if __name__ == "__main__":
    asyncio.run(main())
