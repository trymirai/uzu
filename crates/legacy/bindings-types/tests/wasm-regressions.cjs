// Build the wasm-regressions example, run wasm-bindgen --target nodejs on it,
// then pass the generated .js path to this script.
const assert = require('node:assert/strict');
const path = require('node:path');
const filename = path.resolve(process.argv[2]);
const api = require(filename);
delete require.cache[require.resolve(filename)];
const foreign = require(filename); // Independently instantiated WASM memory.

const record = new api.Record('original');
assert.equal(api.read_record(record), 'original');
assert.throws(() => { record.name = 42; }, TypeError);
assert.equal(record.name, 'original');
record.name = 'updated';
assert.equal(api.read_record(record), 'updated');

const method = '__uzuCloneRecord';
for (const invalid of [0, 1, -1, NaN, Infinity, {}, null, undefined]) {
    assert.throws(() => api.read_record({ [method]: () => invalid }), TypeError);
}
assert.throws(() => api.read_record(new foreign.Record('foreign')), TypeError);
const other = new api.Other('wrong type');
assert.throws(() => api.read_record({ [method]: () => other.__uzuCloneOther() }), TypeError);
assert.equal(record[method](), undefined); // No orphaned clone outside a conversion.

let stale;
assert.equal(api.read_record({ [method]() {
    stale = record[method]();
    return stale;
} }), 'updated');
assert.throws(() => api.read_record({ [method]: () => stale }), TypeError);
assert.throws(() => api.read_record({ [method]() {
    record[method]();
    throw new Error('failed clone');
} }), /failed clone/);
assert.equal(api.read_record({ [method]() {
    const token = record[method]();
    assert.equal(api.read_record(record), 'updated'); // Reentrant conversion.
    return token;
} }), 'updated');

for (const value of ['text', { nested: true }]) {
    const input = { ['__proto__']: value, constructor: 'ctor', toString: 'string' };
    const output = api.roundtrip_map(input);
    assert.equal(Object.getPrototypeOf(output), Object.prototype);
    assert.deepEqual(Object.keys(output).sort(), Object.keys(input).sort());
    assert.deepEqual(output, input);
    assert.deepEqual(api.roundtrip_map(output), input);
}

const choice = new api.ChoiceText('text');
assert.throws(() => { choice.value = 42; }, TypeError);
assert.equal(choice.value, 'text');
choice.value = 'changed';
assert.equal(api.read_choice(choice), 'changed');
assert.equal(api.read_choice(new api.ChoiceEmpty()), '');

console.log('WASM conversion regressions passed');
