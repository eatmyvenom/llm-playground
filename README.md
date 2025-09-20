# llm-monorepo

Monorepo skeleton for experimenting with LangChain-driven agents and a NestJS-based tools server. Built with Turborepo, pnpm, and SWC for fast iteration.

## Prerequisites

- Node.js 20+
- pnpm 9+
- An OpenAI API key (for the LangChain app)

## Getting Started

```bash
pnpm install
pnpm turbo run dev
```

### Environment

Create an `.env` file in `apps/langchain-app` with:

```
OPENAI_API_KEY=sk-...
```

### Common Commands

```bash
pnpm turbo run build      # Build both apps with SWC
pnpm turbo run lint       # Lint via shared ESLint flat config
pnpm turbo run type-check # Run TypeScript in noEmit mode
pnpm --filter langchain-app run dev
pnpm --filter tools-server run dev
```

### Process Management

To run both apps with PM2 in production mode:

```bash
pnpm pm2 start ecosystem.config.js
```

Swagger UI is served at `http://localhost:4000/api` once the tools server is running.

The LangChain app prints the ReAct agent response to stdout.
