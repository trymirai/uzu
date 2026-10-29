import asyncio

from uzu import (
    DecisionAnswer,
    DecisionInput,
    DecisionOption,
    DecisionQuestion,
    Engine,
    EngineConfig,
)


async def main() -> None:
    engine_config = EngineConfig.create()
    engine = await Engine.create(engine_config)

    model = await engine.model("trymirai/d3-flash-M")
    if model is None:
        raise RuntimeError("Model not found")
    async for update in (await engine.download(model)).iterator():
        print(f"\rDownload progress: {update.progress:.2%}", end="", flush=True)
    print()

    session = await engine.decision(model)
    questions = [
        DecisionQuestion.Choice(
            instructions="Which team should handle this request?",
            options=[
                DecisionOption("returns", "Refunds, replacements and damaged deliveries"),
                DecisionOption("billing", "Payments, invoices and charges"),
                DecisionOption("technical", "Product setup and faults"),
            ],
        ),
        DecisionQuestion.YesNo(
            instructions="Does the customer have a receipt?",
            no_description=None,
            yes_description=None,
        ),
        DecisionQuestion.Score(
            instructions="How urgent is this request?",
            levels=["Routine", "Soon", "Today"],
        ),
    ]
    output = await session.decide(
        DecisionInput(
            state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
            questions=questions,
        ),
    )

    for question, result in zip(questions, output.results, strict=True):
        match question, result.answer:
            case DecisionQuestion.Choice(options=options), DecisionAnswer.Choice(key=key, probabilities=probabilities):
                print(f"choice: {key}")
                for option, probability in zip(options, probabilities, strict=True):
                    print(f"  {option.key:<10} {probability:.4f}")
            case DecisionQuestion.Score(levels=levels), DecisionAnswer.Score(score=score, probabilities=probabilities):
                print(f"score: {score:.4f}")
                for level, probability in zip(levels, probabilities, strict=True):
                    print(f"  {level:<10} {probability:.4f}")
            case _, DecisionAnswer.YesNo(yes_probability=yes_probability):
                print(f"yes/no: {yes_probability:.4f}")
        print(f"  {result.stats.tokens_count} tokens in {result.stats.total_duration:.3f} s")


if __name__ == "__main__":
    asyncio.run(main())
