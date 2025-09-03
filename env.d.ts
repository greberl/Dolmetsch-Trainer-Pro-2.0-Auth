// This file augments the NodeJS global namespace to include the API_KEY property
// on process.env. This provides type safety for `process.env.API_KEY`
// used in `vite.config.ts`.

// By using `declare global`, we ensure that we are modifying the global
// NodeJS namespace, rather than creating a new one in the scope of this module.
// `export {}` is added to explicitly mark this file as a module, which is a common
// requirement for global augmentations to be applied correctly.
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      readonly API_KEY: string;
    }
  }
}

export {};
