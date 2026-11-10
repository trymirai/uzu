import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
    root: 'examples',
    resolve: {
        alias: {
            '@trymirai/uzu-wasm': fileURLToPath(new URL('./src/index.ts', import.meta.url)),
        },
    },
});
