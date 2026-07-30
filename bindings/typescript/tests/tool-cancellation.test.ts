import * as z from 'zod';

type InvokeJson = (argumentsJson: string, invocationId: string) => Promise<string>;
type Cancel = (invocationId: string) => void;

jest.mock('../src/napi/index', () => ({
    NativeTool: class {
        readonly testInvokeJson: InvokeJson;
        readonly testCancel: Cancel;

        constructor(_definition: unknown, invokeJson: InvokeJson, cancel: Cancel) {
            this.testInvokeJson = invokeJson;
            this.testCancel = cancel;
        }
    },
    ToolFunction: class {},
    Value: class {},
}));

import { uzuToolFunction } from '../src/tool';

interface TestNativeTool {
    readonly testInvokeJson: InvokeJson;
    readonly testCancel: Cancel;
}

test('retains pre-start cancellations but expires late cancellations', async () => {
    jest.useFakeTimers();
    const tool = uzuToolFunction({
        name: 'is_cancelled',
        parameters: z.object({}),
        returns: z.boolean(),
        handler: (_parameters, context) => context.signal.aborted,
    }) as unknown as TestNativeTool;

    tool.testCancel('before-start');
    await expect(tool.testInvokeJson('{}', 'before-start')).resolves.toBe('true');

    await expect(tool.testInvokeJson('{}', 'already-finished')).resolves.toBe('false');
    tool.testCancel('already-finished');
    jest.runOnlyPendingTimers();

    await expect(tool.testInvokeJson('{}', 'already-finished')).resolves.toBe('false');
    jest.useRealTimers();
});
