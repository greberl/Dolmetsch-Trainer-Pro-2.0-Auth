// This file augments the NodeJS global namespace to include the API_KEY property
// on process.env. This provides type safety for `process.env.API_KEY`
// which is made available to client-side code via Vite's `define` config.

// FIX: Converted to a module with `declare global`. This is a more robust way
// to augment global types and can prevent conflicts with other global declarations
// that may have been causing the `process.cwd()` type error in `vite.config.ts`.
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      readonly API_KEY: string;
    }
  }
}

export {};