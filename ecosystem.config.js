// PM2 process definitions for monorepo services.
// For local development, prefer \`pnpm turbo run dev\` or nodemon/tsx watchers.
// For containerized deployments, consider adding a Docker Compose file.

export default {
  apps: [
    {
      name: "langchain-app",
      cwd: "./apps/langchain-app",
      script: "pnpm",
      args: "run start",
      env: {
        NODE_ENV: "production"
      }
    },
    {
      name: "tools-server",
      cwd: "./apps/tools-server",
      script: "pnpm",
      args: "run start",
      env: {
        NODE_ENV: "production"
      }
    }
  ]
};
