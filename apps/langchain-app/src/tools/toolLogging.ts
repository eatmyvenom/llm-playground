import { logger } from "../logger.js";

export function createToolLogger(toolName: string) {
  return logger.child(`tool:${toolName}`);
}

export function createPreview(raw: string, maxLength = 200): string {
  const normalized = raw.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return "";
  }

  if (normalized.length <= maxLength) {
    return normalized;
  }

  const limit = Math.max(0, maxLength - 3);
  return `${normalized.slice(0, limit)}...`;
}
