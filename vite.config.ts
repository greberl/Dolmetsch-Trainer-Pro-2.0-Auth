/// <reference types="node" />
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
// Fix: The 'process' module does not have a named export 'cwd'.
// Instead, use the `process.cwd()` method from the global `process` object,
// which is available in the Node.js environment where Vite config is run.
// The `/// <reference types="node" />` directive helps TypeScript recognize it.

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
  const env = loadEnv(mode, process.cwd(), '');
  return {
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify(env.API_KEY),
    },
  }
})