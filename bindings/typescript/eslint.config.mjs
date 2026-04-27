// @ts-check
import prettier from 'eslint-plugin-prettier';
import unusedImports from 'eslint-plugin-unused-imports';
import tseslint from 'typescript-eslint';

export default tseslint.config(
    {
        languageOptions: {
            parser: tseslint.parser,
            parserOptions: { sourceType: 'module' },
        },
        files: ['**/*.ts', '**/*.mts', '**/*.cts', '**/*.js', '**/*.mjs', '**/*.cjs'],
        ignores: ['dist/'],
        plugins: {
            '@typescript-eslint': tseslint.plugin,
            'unused-imports': unusedImports,
            prettier,
        },
        rules: {
            'no-unused-vars': 'off',
            'prettier/prettier': 'error',
            'unused-imports/no-unused-imports': 'error',
            'no-restricted-imports': [
                'error',
                {
                    patterns: [
                        {
                            regex: '^@trymirai/uzu(/.*)?',
                            message: 'Use a relative import, not a package import.',
                        },
                    ],
                },
            ],
        },
    },
    {
        files: ['tests/**', 'examples/**'],
        rules: {
            'no-restricted-imports': 'off',
        },
    },
);
