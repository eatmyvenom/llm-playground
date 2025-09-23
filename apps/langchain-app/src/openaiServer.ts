import "./env.js";
import Fastify from "fastify";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { randomUUID } from "node:crypto";
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import type { Logger } from "@llm/logger";
import { buildPlanningAgent } from "./agents/planningExecutorAgent.js";
import { buildReactAgent } from "./agents/reactAgent.js";
import { logger } from "./logger.js";

const planningAgent = buildPlanningAgent();

interface ChatCompletionRequest {
  model: string;
  messages: Array<ChatCompletionMessage>;
  stream?: boolean;
}

interface ChatCompletionMessage {
  role: string;
  content: string | Array<TextContentBlock>;
}

interface TextContentBlock {
  type: "text";
  text: string;
}

interface ChatCompletionError {
  error: {
    message: string;
    type: string;
    param: null;
    code: string | null;
  };
}

type ChatCompletionResponse = {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: "assistant"; content: string };
    finish_reason: "stop";
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

async function createServer(): Promise<FastifyInstance> {
  const fastify = Fastify();
  const reactAgent = await buildReactAgent();

  fastify.get("/health", async () => ({ status: "ok" }));

  fastify.post(
    "/v1/chat/completions",
    async (request: FastifyRequest<{ Body: ChatCompletionRequest }>, reply: FastifyReply) => {
      const requestId = randomUUID();
      const requestLogger = logger.child(`openai:${requestId}`);
      const startedAt = Date.now();

      try {
        const body = request.body;
        if (!body || typeof body !== "object") {
          requestLogger.warn("Rejected request with non-object body", typeof body);
          return reply.status(400).send(buildError("Request body must be a JSON object."));
        }

        if (!body.model) {
          requestLogger.warn("Rejected request with missing model field");
          return reply.status(400).send(buildError("Missing required field: model."));
        }

        if (!Array.isArray(body.messages) || body.messages.length === 0) {
          requestLogger.warn("Rejected request with invalid messages payload", body.model);
          return reply.status(400).send(buildError("messages must be a non-empty array."));
        }

        if (body.stream) {
          requestLogger.warn("Rejected request with unsupported streaming flag", body.model);
          return reply.status(400).send(buildError("stream=true is not supported for this endpoint."));
        }

        requestLogger.info("Processing chat completion request", describeRequest(body));

        switch (body.model) {
          case "planning-agent": {
            const content = await handlePlanningRequest(body.messages, requestLogger);
            requestLogger.info("Planning agent completed", {
              durationMs: Date.now() - startedAt,
              completionPreview: createPreview(content),
            });
            return reply.send(buildResponse(body.model, content));
          }
          case "react-agent": {
            const content = await handleReactRequest(body.messages, reactAgent, requestLogger);
            requestLogger.info("ReAct agent completed", {
              durationMs: Date.now() - startedAt,
              completionPreview: createPreview(content),
            });
            return reply.send(buildResponse(body.model, content));
          }
          default:
            requestLogger.warn("Rejected request for unknown model", body.model);
            return reply.status(400).send(buildError(`Unknown model: ${body.model}`));
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unexpected error";
        requestLogger.error("Failed to handle chat completion request", error);
        return reply.status(500).send(buildError(message, "internal_server_error"));
      }
    },
  );

  return fastify;
}

async function handlePlanningRequest(messages: Array<ChatCompletionMessage>, requestLogger: Logger): Promise<string> {
  const task = collectUserContent(messages).trim();
  if (!task) {
    throw new Error("planning-agent requires at least one user message with text content.");
  }

  requestLogger.info("Planning agent started", {
    taskPreview: createPreview(task),
  });

  const stream = planningAgent.streamTask(task);
  let final = "";
  let chunkIndex = 0;
  for await (const chunk of stream) {
    final += chunk;
    const preview = createPreview(chunk);
    if (preview) {
      requestLogger.debug("Planning agent chunk", {
        chunkIndex,
        preview,
      });
    }
    chunkIndex += 1;
  }

  requestLogger.info("Planning agent finished", {
    chunkCount: chunkIndex,
  });

  return final;
}

async function handleReactRequest(
  messages: Array<ChatCompletionMessage>,
  reactAgent: Awaited<ReturnType<typeof buildReactAgent>>,
  requestLogger: Logger,
): Promise<string> {
  requestLogger.info("ReAct agent started", {
    messageCount: messages.length,
    userPreview: createPreview(collectUserContent(messages)),
  });

  const langchainMessages = toLangchainMessages(messages);
  const response = await reactAgent.invoke({ messages: langchainMessages });
  const finalMessage = response.messages.at(-1);
  if (!finalMessage) {
    throw new Error("ReAct agent did not return any messages.");
  }

  if (Array.isArray(response.messages)) {
    response.messages.forEach((message, index) => {
      const summary = describeLangchainMessage(message);
      if (summary) {
        requestLogger.debug("ReAct agent message", {
          index,
          ...summary,
        });
      }
    });
  }

  const content = finalMessage.content;
  if (typeof content === "string") {
    requestLogger.info("ReAct agent finished", {
      finalPreview: createPreview(content),
    });
    return content;
  }

  if (Array.isArray(content)) {
    const joined = content
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item.type === "text") {
          return item.text;
        }
        return "";
      })
      .join("");
    requestLogger.info("ReAct agent finished", {
      finalPreview: createPreview(joined),
    });
    return joined;
  }

  const serialized = JSON.stringify(content);
  requestLogger.info("ReAct agent finished", {
    finalPreview: createPreview(serialized),
  });
  return serialized;
}

function collectUserContent(messages: Array<ChatCompletionMessage>): string {
  return messages
    .filter((message) => message.role === "user")
    .map((message) => extractTextContent(message.content))
    .filter(Boolean)
    .join("\n\n");
}

function toLangchainMessages(messages: Array<ChatCompletionMessage>): Array<BaseMessage> {
  return messages.map((message) => {
    const content = extractTextContent(message.content);
    switch (message.role) {
      case "system":
        return new SystemMessage(content);
      case "assistant":
        return new AIMessage(content);
      case "tool":
        return new ToolMessage({ content, tool_call_id: "tool_call_response" });
      default:
        return new HumanMessage(content);
    }
  });
}

function extractTextContent(content: ChatCompletionMessage["content"]): string {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .filter((item): item is TextContentBlock => item?.type === "text" && typeof item.text === "string")
    .map((item) => item.text)
    .join("");
}

function describeRequest(body: ChatCompletionRequest): Record<string, unknown> {
  return {
    model: body.model,
    messageCount: body.messages.length,
    userPreview: createPreview(collectUserContent(body.messages)),
  };
}

function buildResponse(model: string, content: string): ChatCompletionResponse {
  return {
    id: `chatcmpl-${randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
  };
}

function buildError(message: string, type: string = "invalid_request_error"): ChatCompletionError {
  return {
    error: {
      message,
      type,
      param: null,
      code: null,
    },
  };
}

function createPreview(raw: string, maxLength = 200): string {
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

function describeLangchainMessage(message: BaseMessage): { type: string; preview: string } | null {
  const type = inferMessageType(message);
  const flattened = flattenLangChainContent(message.content);
  const preview = createPreview(flattened);

  if (!preview) {
    return { type, preview: "[no content]" };
  }

  return { type, preview };
}

function inferMessageType(message: BaseMessage): string {
  const candidate = (message as { _getType?: () => string })._getType;
  if (typeof candidate === "function") {
    try {
      return candidate.call(message);
    } catch {
      // fall through to constructor-based inference
    }
  }

  return message.constructor?.name ?? "unknown";
}

function flattenLangChainContent(content: BaseMessage["content"]): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object") {
          if ("type" in item && (item as { type?: unknown }).type === "text") {
            const text = (item as { text?: unknown }).text;
            return typeof text === "string" ? text : "";
          }

          try {
            return JSON.stringify(item);
          } catch {
            return "[unserializable]";
          }
        }
        return "";
      })
      .join("");
  }

  if (content && typeof content === "object") {
    try {
      return JSON.stringify(content);
    } catch {
      return "[unserializable]";
    }
  }

  return "";
}

async function main(): Promise<void> {
  try {
    const server = await createServer();
    const port = process.env.OPENAI_AGENT_PORT ? Number(process.env.OPENAI_AGENT_PORT) : 5050;
    await server.listen({ port, host: "0.0.0.0" });
    logger.info(`OpenAI-compatible API listening on http://localhost:${port}`);
  } catch (error) {
    logger.error("Failed to start OpenAI-compatible server", error);
    process.exitCode = 1;
  }
}

void main();
