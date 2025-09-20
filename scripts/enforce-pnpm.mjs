#!/usr/bin/env node
/**
 * Simple enforcement script to ensure pnpm is used for installing dependencies.
 */
const execPath = process.env.npm_execpath || "";
if (!execPath.includes("pnpm")) {
  console.error("\nThis repository requires pnpm for dependency management.\nInstall pnpm and run pnpm install instead.\n");
  process.exit(1);
}
