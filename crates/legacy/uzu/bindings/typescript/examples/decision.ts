import {
    DecisionAnswerChoice,
    DecisionAnswerScore,
    DecisionAnswerYesNo,
    DecisionInput,
    DecisionOption,
    DecisionQuestionChoice,
    DecisionQuestionScore,
    DecisionQuestionYesNo,
    Engine,
    EngineConfig,
} from '@trymirai/uzu';

async function main() {
    let engineConfig = EngineConfig.create();
    let engine = await Engine.create(engineConfig);

    let model = await engine.model('trymirai/d3-flash-M');
    if (!model) {
        throw new Error('Model not found');
    }
    for await (const update of await engine.download(model)) {
        process.stdout.write(`\rDownload progress: ${(update.progress * 100).toFixed(2)}%`);
    }
    console.log();

    let session = await engine.decision(model);
    let questions = [
        new DecisionQuestionChoice('Which team should handle this request?', [
            new DecisionOption('returns', 'Refunds, replacements and damaged deliveries'),
            new DecisionOption('billing', 'Payments, invoices and charges'),
            new DecisionOption('technical', 'Product setup and faults'),
        ]),
        new DecisionQuestionYesNo('Does the customer have a receipt?'),
        new DecisionQuestionScore('How urgent is this request?', ['Routine', 'Soon', 'Today']),
    ];
    let output = await session.decide(
        new DecisionInput(
            'The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.',
            questions,
        ),
    );

    for (const [index, result] of output.results.entries()) {
        let question = questions[index];
        let answer = result.answer;
        if (question instanceof DecisionQuestionChoice && answer instanceof DecisionAnswerChoice) {
            console.log(`choice: ${answer.key}`);
            for (const [optionIndex, option] of question.options.entries()) {
                console.log(`  ${option.key.padEnd(10)} ${answer.probabilities[optionIndex]?.toFixed(4)}`);
            }
        } else if (question instanceof DecisionQuestionScore && answer instanceof DecisionAnswerScore) {
            console.log(`score: ${answer.score.toFixed(4)}`);
            for (const [levelIndex, level] of question.levels.entries()) {
                console.log(`  ${level.padEnd(10)} ${answer.probabilities[levelIndex]?.toFixed(4)}`);
            }
        } else if (answer instanceof DecisionAnswerYesNo) {
            console.log(`yes/no: ${answer.yesProbability.toFixed(4)}`);
        }
        console.log(`  ${result.stats.tokensCount} tokens in ${result.stats.totalDuration.toFixed(3)} s`);
    }
}

main().catch((error) => {
    console.error(error);
});
