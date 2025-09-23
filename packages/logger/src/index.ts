import { createWriteStream, existsSync, mkdirSync } from "node:fs";
import type { WriteStream } from "node:fs";
import { join, resolve } from "node:path";

export type LogLevel = "debug" | "info" | "warn" | "error";

export interface LoggerOptions {
  readonly name?: string;
  readonly level?: LogLevel;
}

export interface Logger {
  readonly level: LogLevel;
  debug: (...args: LogArguments) => void;
  info: (...args: LogArguments) => void;
  warn: (...args: LogArguments) => void;
  error: (...args: LogArguments) => void;
  child: (name: string) => Logger;
}

interface FileSinks {
  readonly warn: WriteStream;
  readonly error: WriteStream;
}

type LogArguments = [message: unknown, ...meta: unknown[]];

type LevelWeight = Record<LogLevel, number>;

const LEVEL_WEIGHTS: LevelWeight = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

const DEFAULT_LEVEL: LogLevel = parseLevel(process.env.LOG_LEVEL) ?? inferDefaultLevel();

let cachedSinks: FileSinks | null = null;
let processHandlersInstalled = false;

export function createLogger(options: LoggerOptions = {}): Logger {
  const context = options.name?.trim() ?? "";
  const level = options.level ?? DEFAULT_LEVEL;

  const emit = (levelToEmit: LogLevel, args: LogArguments): void => {
    if (!shouldLog(level, levelToEmit)) {
      return;
    }

    const [message, ...meta] = args;
    const line = formatLine(levelToEmit, context, message, meta);
    outputToConsole(levelToEmit, line);
    writeToFile(levelToEmit, line);
  };

  const child = (name: string): Logger => {
    const trimmed = name.trim();
    const childName = trimmed.length > 0 ? joinContext(context, trimmed) : context;
    return createLogger({ ...options, name: childName, level });
  };

  return {
    level,
    debug: (...args) => emit("debug", args),
    info: (...args) => emit("info", args),
    warn: (...args) => emit("warn", args),
    error: (...args) => emit("error", args),
    child,
  };
}

export function installProcessHandlers(logger: Logger): void {
  if (processHandlersInstalled) {
    return;
  }

  processHandlersInstalled = true;

  process.on("unhandledRejection", (reason) => {
    logger.error("Unhandled promise rejection", reason);
  });

  process.on("uncaughtException", (error) => {
    logger.error("Uncaught exception", error);
  });
}

function shouldLog(currentLevel: LogLevel, candidate: LogLevel): boolean {
  return LEVEL_WEIGHTS[candidate] >= LEVEL_WEIGHTS[currentLevel];
}

function parseLevel(candidate: string | undefined | null): LogLevel | null {
  if (!candidate) {
    return null;
  }

  const normalized = candidate.toLowerCase();
  if (normalized === "debug" || normalized === "info" || normalized === "warn" || normalized === "error") {
    return normalized;
  }

  return null;
}

function inferDefaultLevel(): LogLevel {
  return process.env.NODE_ENV === "production" ? "info" : "debug";
}

function joinContext(parent: string, child: string): string {
  if (!parent) {
    return child;
  }

  return `${parent}:${child}`;
}

function formatLine(level: LogLevel, context: string, message: unknown, meta: unknown[]): string {
  const timestamp = new Date().toISOString();
  const levelLabel = level.toUpperCase();
  const contextLabel = context ? ` [${context}]` : "";
  const body = formatMessage(message);
  const trailing = meta.length > 0 ? ` ${meta.map(formatMessage).join(" ")}` : "";

  return `${timestamp} ${levelLabel}${contextLabel} ${body}${trailing}`;
}

function formatMessage(value: unknown): string {
  if (value instanceof Error) {
    return value.stack ?? value.message;
  }

  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") {
    return String(value);
  }

  if (typeof value === "undefined") {
    return "undefined";
  }

  if (value === null) {
    return "null";
  }

  try {
    return JSON.stringify(value);
  } catch (error) {
    return `[unserializable:${(error as Error).message}]`;
  }
}

function outputToConsole(level: LogLevel, line: string): void {
  switch (level) {
    case "error":
      console.error(line);
      break;
    case "warn":
      console.warn(line);
      break;
    default:
      console.log(line);
  }
}

function writeToFile(level: LogLevel, line: string): void {
  if (level !== "warn" && level !== "error") {
    return;
  }

  const sinks = getSinks();

  if (level === "warn") {
    sinks.warn.write(`${line}\n`);
  } else {
    sinks.error.write(`${line}\n`);
  }
}

function getSinks(): FileSinks {
  if (cachedSinks) {
    return cachedSinks;
  }

  const directory = process.env.LOG_DIR ? resolve(process.env.LOG_DIR) : resolve(process.cwd(), "logs");

  if (!existsSync(directory)) {
    mkdirSync(directory, { recursive: true });
  }

  const envName = process.env.NODE_ENV ?? "development";
  const warnPath = join(directory, `${envName}.warn.log`);
  const errorPath = join(directory, `${envName}.error.log`);

  cachedSinks = {
    warn: createWriteStream(warnPath, { flags: "a" }),
    error: createWriteStream(errorPath, { flags: "a" }),
  };

  return cachedSinks;
}
