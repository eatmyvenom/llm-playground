# Running the llm-monorepo

This runbook walks through setting up the workspace, configuring environment variables, and running both applications in development and production-style modes.

## 1. Prepare Your Environment
1. Install Node.js 20 or newer. Using [corepack](https://nodejs.org/docs/latest-v20.x/api/corepack.html) keeps pnpm aligned with the repo requirements.
2. Enable pnpm via Corepack (recommended):
   ```bash
   corepack enable pnpm
   corepack prepare pnpm@9 --activate
   ```
3. Verify pnpm is available and at least v9:
   ```bash
   pnpm --version
   ```
   The repository ships with `scripts/enforce-pnpm.mjs`, which prevents installs with npm or yarn.

## 2. Install Dependencies
1. From the repository root, install the workspace dependencies:
   ```bash
   pnpm install
   ```
2. The Turborepo cache will populate after the first command run. Subsequent `pnpm turbo` commands will be faster.

## 3. Configure Environment Variables
1. Copy the repository’s `.env.example` to `.env` at the project root and replace the placeholder values:
   ```bash
   cp .env.example .env
   ```
   At minimum set `OPENAI_API_KEY`; the example file also lists `TOOLS_SERVER_URL`, `PORT`, and the `PLANNING_AGENT_*` overrides.
2. Both apps load the root `.env` automatically via `apps/langchain-app/src/env.ts` and `apps/tools-server/src/env.ts`, so you do not need per-package `.env` files.
3. If you want package-specific overrides, drop an `.env` next to the package `package.json`; those values override the shared file.
4. Optional environment variables you can set at runtime:
   - `TOOLS_SERVER_URL`: Base URL the agents use to reach the MCP tools server (defaults to `http://localhost:4000`).
   - `PLANNING_AGENT_PLANNER_MODEL`, `PLANNING_AGENT_EXECUTOR_MODEL`, `PLANNING_AGENT_RESPONDER_MODEL`: Override the default OpenAI models for the planning agent pipeline.
   - `OPENAI_AGENT_PORT`: Used by Turborepo when you run agent-specific dev servers.

## 4. Run Both Apps in Development
1. Launch both watch-mode processes from the monorepo root:
   ```bash
   pnpm turbo run dev
   ```
2. Expected behavior:
   - `apps/tools-server`: Starts on `http://localhost:4000`, applies validation pipes, and exposes Swagger UI at `http://localhost:4000/api` plus MCP-style routes under `/mcp`.
   - `apps/langchain-app`: Streams planning output to stdout, then runs the ReAct agent once the tools server is reachable.
3. Stop the processes with `Ctrl+C`. Turborepo forwards the signal to both child processes.

### Run Packages Individually (Optional)
- Agent only:
  ```bash
  pnpm --filter langchain-app run dev
  ```
- Tools server only:
  ```bash
  pnpm --filter tools-server run dev
  ```
  When running separately, set `TOOLS_SERVER_URL=http://localhost:4000` for the agent process if it cannot detect the server automatically.

## 5. Exercise the Agent Workflow
1. Start both apps (see Step 4).
2. Watch the terminal output from `langchain-app`:
   - The planning agent streams multiple messages describing the long-horizon plan.
   - The ReAct agent invokes the MCP web search tool and prints its final answer.
3. Invoke the tools server directly if needed:
   ```bash
   curl -X POST \
     -H 'content-type: application/json' \
     -d '{"jsonrpc":"2.0","id":"dev"}' \
     http://localhost:4000/mcp/tools/list
   ```
   or inspect request/response shapes via Swagger at `/api`.
4. Edit source files; the `tsx` watchers rebuild automatically.

## 6. Build for Production-like Runs
1. Compile both packages with SWC:
   ```bash
   pnpm turbo run build
   ```
2. Start the compiled services with PM2:
   ```bash
   pnpm pm2 start ecosystem.config.js
   ```
3. Inspect status and logs:
   ```bash
   pnpm pm2 ls
   pnpm pm2 logs
   ```
4. Stop the processes when finished:
   ```bash
   pnpm pm2 stop all
   pnpm pm2 delete all
   ```
   You can also run each package manually via `pnpm --filter <package> run start`.

## 7. Inspect Logs in Dev and Production
1. Both applications emit structured logs through the shared `@llm/logger` package.
2. Warnings and errors are persisted under `logs/` with filenames based on `NODE_ENV` (for example, `logs/development.warn.log` and `logs/development.error.log`).
3. Override the output directory by setting `LOG_DIR`; if unset, the directory is created alongside the process working directory.
4. Control verbosity with `LOG_LEVEL` (`debug`, `info`, `warn`, `error`). Production defaults to `info`, development defaults to `debug`.
5. Unhandled promise rejections and uncaught exceptions are automatically captured and written to the error log.

## 8. Quality Gates Before Shipping Changes
1. Lint all workspaces:
   ```bash
   pnpm turbo run lint
   ```
2. Run TypeScript type-checking:
   ```bash
   pnpm turbo run type-check
   ```
3. Format code when necessary:
   ```bash
   pnpm format
   ```
4. Clean build outputs if the cache becomes stale:
   ```bash
   pnpm turbo run clean --filter=...
   ```

## 9. Troubleshooting Tips
- **Missing `OPENAI_API_KEY`**: The ReAct agent throws an error immediately; confirm the `.env` file is loaded or export the variable in your shell.
- **Tools server connection errors**: Ensure the server is running on the same port specified by `TOOLS_SERVER_URL`. Cross-check with `curl http://localhost:4000/health` if you expose a health endpoint.
- **Permission errors on install**: Verify you are using pnpm v9+ and not running in a restricted directory.
- **Model overrides**: If you override planner/executor/responder models, pick model IDs supported by your OpenAI account; otherwise requests will fail at runtime.

Following the sequence above guarantees both services compile, pass checks, and run end-to-end with OpenAI tooling connected through the MCP bridge.
