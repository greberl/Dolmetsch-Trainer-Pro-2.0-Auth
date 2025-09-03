/// <reference types="node" />
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
// Fix: `cwd` is not a named export from `node:process`. Import `process` itself
// to resolve TypeScript errors like "Property 'cwd' does not exist on type 'Process'".
// This avoids ambiguity with the global `process` object's type, which can be
// problematic in projects that contain both Node.js and browser-targeted code.
import process from 'node:process';

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
