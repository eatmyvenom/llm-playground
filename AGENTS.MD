# Agent Operations Guide

This repository hosts a LangChain-based agent application and a NestJS tools server that exposes MCP-style tooling. Follow this guide whenever you use an LLM or autonomous agent to modify or run the project.

## Monorepo Shape
- `apps/langchain-app`: TypeScript ReAct agent built on LangChain, uses `ChatOpenAI` (`gpt-4.1-mini`).
- `apps/tools-server`: NestJS REST + Swagger API that mirrors MCP `tools/list` and `tools/call` endpoints. Houses reusable tool classes.
- `packages/eslint-config`: Shared flat-config ESLint preset (`@llm/eslint-config`).
- `packages/typescript-config`: Base `tsconfig` used across apps (`module`/`moduleResolution` are `NodeNext`).
- `scripts/enforce-pnpm.mjs`: Blocks `npm/yarn` installs; always use pnpm.

## Required Tooling
- Node.js 20+ and pnpm 9+ (checked via `package.json` engines and the preinstall script).
- Install once with `pnpm install` (root). All workspace commands should be run through pnpm.
- Turborepo orchestrates tasks; prefer the composite commands below instead of invoking per-package binaries directly:
  - `pnpm turbo run dev` – runs both apps in watch mode (`tsx`).
  - `pnpm turbo run build` – compiles via SWC into each package’s `dist/`.
  - `pnpm turbo run lint` / `type-check` – shared lint + TS checks.
- Format with `pnpm format` (Prettier over `ts/tsx/js/json/md`).
- For production-like runs, use `pnpm pm2 start ecosystem.config.js` (launches both apps from `dist/`).

## Environment & Secrets
- `apps/langchain-app` requires `OPENAI_API_KEY` (set in `apps/langchain-app/.env`). Without it `buildReactAgent` throws.
- The tools server listens on `PORT` (default 4000) and serves Swagger at `/api`.
- Do not commit secret values; rely on `.env`, process managers, or deployment configuration.

## Coding Standards
- All runtime code is TypeScript targeting ES2022; stay within the existing module (`NodeNext`) and ESM setup. Avoid introducing CommonJS.
- Reuse shared configs:
  - Import ESLint config with `createConfig({ tsconfigPath: "./tsconfig.json" })` when adding packages.
  - Extend `../../packages/typescript-config/tsconfig.json` for new `tsconfig`s.
- Lint before finishing: `pnpm turbo run lint`. Fix diagnostics instead of disabling rules. Key rules enforce consistent type imports, explicit returns, and safe promise handling.
- Prefer lightweight comments; use them only for non-obvious context (mirrors repository style).
- Keep dependencies aligned with pnpm overrides (`langchain@next`, etc.). Add deps via `pnpm add --filter <pkg>`.

## Agent Patterns
- LangChain tools live in `apps/langchain-app/src/agents/`. Each tool uses `DynamicStructuredTool` with `zod` schemas. When adding tools, provide precise descriptions and enforce lowercase IDs where relevant.
- The ReAct agent currently injects one tool (`query_mock_records`). Extend the `tools` array inside `buildReactAgent()`; keep temperature low for deterministic tests.
- MCP bridging lives in `apps/tools-server/src/mcp/`. New tools should:
  1. Implement a class similar to `SampleDatabaseTool` with `listDefinition()` and `call()`.
  2. Register the class in `AppModule` providers and inject into `McpService`.
  3. Update `McpService.listTools()` and `callTool()` to expose the new tool.
- REST DTOs leverage `class-validator` and `class-transformer`; keep payload validation strict (`whitelist: true` in `main.ts`).

## Testing & Verification
- Run `pnpm turbo run lint` and `pnpm turbo run type-check` prior to submitting changes. No automated test suite exists yet—unit coverage is a gap, so add tests (Jest or Vitest) when introducing non-trivial logic.
- For manual validation:
  - `pnpm --filter langchain-app run dev` prints agent outputs in the terminal.
  - `pnpm --filter tools-server run dev` exposes Swagger at `http://localhost:4000/api` and MCP endpoints under `/mcp`.
- Keep generated `dist/` folders out of version control; builds should be reproducible.

## Operational Notes
- Logging is console-based; avoid leaving noisy debug output.
- When adjusting build pipelines, edit `turbo.json` so dependent tasks stay consistent.
- If you introduce long-running background work, ensure PM2 config reflects the change (`ecosystem.config.js`).

Stick to these conventions to keep LLM-driven contributions predictable and safe across the monorepo.
