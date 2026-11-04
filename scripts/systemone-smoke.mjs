import assert from 'node:assert/strict';
import { choice, noul, score, TypeSafeClient } from '@typesafe-ai/sdk';

const baseURL = process.env.UZU_URL ?? 'http://127.0.0.1:8000';
const modelsResponse = await fetch(new URL('/v1/models', baseURL));
assert.equal(modelsResponse.status, 200);
const models = await modelsResponse.json();
const model = process.env.UZU_MODEL ?? models.data[0].id;
const client = new TypeSafeClient({ baseURL, apiKey: 'local', defaultModel: model,
  compression: false, timeout: 35000, retry: { maxRetries: 0 } });
const result = await client.systemOne({
  state: 'I was charged twice. Please refund the duplicate charge today. ' + 'Customer supplied data. '.repeat(60),
  questions: {
    category: choice('Classify the ticket.', { billing: 'Payments and refunds', technical: 'Technical issues', other: null }),
    urgent: noul('Does the customer ask for action today?'),
    priority: score(null, ['Routine', 'Needs attention this week', 'Needs attention today']),
    singleton: choice(null, { only: null }),
  },
});
assert.equal(result.model, model);
assert.deepEqual(Object.keys(result.answers), ['category', 'urgent', 'priority', 'singleton']);
assert.equal(result.answers.singleton.choice, 'only');
assert.equal(result.answers.singleton.confidence, 1);
assert.equal(result.answers.urgent.type, 'noul');
assert(result.answers.urgent.noul >= 0 && result.answers.urgent.noul <= 1);
assert(result.answers.priority.score >= 0 && result.answers.priority.score <= 2);
for (const name of ['category', 'priority', 'singleton']) {
  const answer = result.answers[name];
  assert(answer.confidence >= 0 && answer.confidence <= 1);
  const sum = Object.values(answer.probabilities).reduce((a, b) => a + b, 0);
  assert(Math.abs(sum - 1) < 1e-9);
}
assert(result.usage.input_tokens > 0);
assert.equal(result.usage.output_tokens, 0);
console.log(JSON.stringify(result, null, 2));
