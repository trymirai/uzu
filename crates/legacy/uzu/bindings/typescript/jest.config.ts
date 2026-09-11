import type { JestConfigWithTsJest } from 'ts-jest';

const config: JestConfigWithTsJest = {
    preset: 'ts-jest/presets/default-esm',
    testEnvironment: 'node',
    transform: {
        '^.+\\.(t|j)sx?$': ['@swc/jest', { sourceMaps: 'inline' }],
    },
    moduleNameMapper: {
        '^@trymirai/uzu$': '<rootDir>/src/index.ts',
        '^@trymirai/uzu/(.*)$': '<rootDir>/src/$1',
    },
    modulePathIgnorePatterns: [
        '<rootDir>/dist/',
        '<rootDir>/deno/',
        '<rootDir>/deno_tests/',
        '<rootDir>/packages/',
    ],
    testPathIgnorePatterns: ['scripts'],
    // prettierPath: require.resolve('prettier-2'),
};

export default config;
