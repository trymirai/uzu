import * as z from 'zod';

import { NativeTool, ToolFunction, Value } from './napi/index';

export interface UzuToolContext {
    readonly signal: AbortSignal;
}

export interface UzuToolInvokeOptions {
    readonly signal?: AbortSignal;
}

export interface UzuToolFunctionOptions<Parameters extends z.ZodObject, ResultSchema extends z.ZodType> {
    readonly name: string;
    readonly description?: string;
    readonly parameters: Parameters;
    readonly returns: ResultSchema;
    readonly handler: (
        parameters: z.output<Parameters>,
        context: UzuToolContext,
    ) => z.input<ResultSchema> | Promise<z.input<ResultSchema>>;
}

export class UzuToolFunction<
    Parameters extends z.ZodObject,
    ResultSchema extends z.ZodType,
> extends NativeTool {
    readonly name: string;
    readonly description: string;
    readonly parameters: Parameters;
    readonly returns: ResultSchema;
    readonly parametersSchema: Record<string, unknown>;
    readonly returnSchema: Record<string, unknown>;
    readonly handler: UzuToolFunctionOptions<Parameters, ResultSchema>['handler'];

    constructor(options: UzuToolFunctionOptions<Parameters, ResultSchema>) {
        const name = options.name.trim();
        if (!name) {
            throw new TypeError('tool name must not be empty');
        }

        const description = options.description ?? '';
        const parametersSchema = z.toJSONSchema(options.parameters);
        const returnSchema = z.toJSONSchema(options.returns);
        const activeInvocations = new Map<string, AbortController>();
        const cancelledInvocations = new Set<string>();

        const invokeJson = async (argumentsJson: string, invocationId: string): Promise<string> => {
            const controller = new AbortController();
            activeInvocations.set(invocationId, controller);
            if (cancelledInvocations.delete(invocationId)) {
                controller.abort();
            }

            try {
                const rawArguments: unknown = JSON.parse(argumentsJson);
                const parameters = await options.parameters.parseAsync(rawArguments);
                const rawResult = await options.handler(parameters, {
                    signal: controller.signal,
                });
                const result = await options.returns.parseAsync(rawResult);
                return serializeResult(result);
            } finally {
                activeInvocations.delete(invocationId);
                cancelledInvocations.delete(invocationId);
            }
        };
        const cancel = (invocationId: string): void => {
            const controller = activeInvocations.get(invocationId);
            if (controller) {
                controller.abort();
            } else {
                cancelledInvocations.add(invocationId);
            }
        };

        super(
            new ToolFunction(
                name,
                description,
                new Value(JSON.stringify(parametersSchema)),
                new Value(JSON.stringify(returnSchema)),
            ),
            invokeJson,
            cancel,
        );

        this.name = name;
        this.description = description;
        this.parameters = options.parameters;
        this.returns = options.returns;
        this.parametersSchema = parametersSchema;
        this.returnSchema = returnSchema;
        this.handler = options.handler;
    }

    async invoke(
        input: z.input<Parameters>,
        options: UzuToolInvokeOptions = {},
    ): Promise<z.output<ResultSchema>> {
        const parameters = await this.parameters.parseAsync(input);
        const signal = options.signal ?? new AbortController().signal;
        const rawResult = await this.handler(parameters, { signal });
        return this.returns.parseAsync(rawResult);
    }
}

export function uzuToolFunction<Parameters extends z.ZodObject, ResultSchema extends z.ZodType>(
    options: UzuToolFunctionOptions<Parameters, ResultSchema>,
): UzuToolFunction<Parameters, ResultSchema> {
    return new UzuToolFunction(options);
}

function serializeResult(result: unknown): string {
    const json = JSON.stringify(result === undefined ? null : result);
    if (json === undefined) {
        throw new TypeError('tool result must be JSON serializable');
    }
    return json;
}
