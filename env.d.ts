// This file augments the NodeJS global namespace to include the API_KEY property
// on process.env. This provides type safety for `process.env.API_KEY`
// which is made available to client-side code via Vite's `define` config.

// FIX: Changed from a module with `declare global` to a global script.
// This avoids module-scoping issues that can cause conflicts with Node.js
// global types and resolves the type error for `process.cwd()` in `vite.config.ts`.
declare namespace NodeJS {
  interface ProcessEnv {
    readonly API_KEY: string;
  }
}
