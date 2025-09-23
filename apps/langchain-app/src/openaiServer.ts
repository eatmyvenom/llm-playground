import "./env.js";
import Fastify from "fastify";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { randomUUID } from "node:crypto";
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";
import { buildPlanningAgent } from "./agents/planningExecutorAgent.js";
import { buildReactAgent } from "./agents/reactAgent.js";

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
      try {
        const body = request.body;
        if (!body || typeof body !== "object") {
          return reply.status(400).send(buildError("Request body must be a JSON object."));
        }

        if (!body.model) {
          return reply.status(400).send(buildError("Missing required field: model."));
        }

        if (!Array.isArray(body.messages) || body.messages.length === 0) {
          return reply.status(400).send(buildError("messages must be a non-empty array."));
        }

        if (body.stream) {
          return reply.status(400).send(buildError("stream=true is not supported for this endpoint."));
        }

        switch (body.model) {
          case "planning-agent": {
            const content = await handlePlanningRequest(body.messages);
            return reply.send(buildResponse(body.model, content));
          }
          case "react-agent": {
            const content = await handleReactRequest(body.messages, reactAgent);
            return reply.send(buildResponse(body.model, content));
          }
          default:
            return reply.status(400).send(buildError(`Unknown model: ${body.model}`));
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unexpected error";
        return reply.status(500).send(buildError(message, "internal_server_error"));
      }
    },
  );

  return fastify;
}

async function handlePlanningRequest(messages: Array<ChatCompletionMessage>): Promise<string> {
  const task = collectUserContent(messages).trim();
  if (!task) {
    throw new Error("planning-agent requires at least one user message with text content.");
  }

  const stream = planningAgent.streamTask(task);
  let final = "";
  for await (const chunk of stream) {
    final += chunk;
  }

  return final;
}

async function handleReactRequest(
  messages: Array<ChatCompletionMessage>,
  reactAgent: Awaited<ReturnType<typeof buildReactAgent>>,
): Promise<string> {
  const langchainMessages = toLangchainMessages(messages);
  const response = await reactAgent.invoke({ messages: langchainMessages });
  const finalMessage = response.messages.at(-1);
  if (!finalMessage) {
    throw new Error("ReAct agent did not return any messages.");
  }

  const content = finalMessage.content;
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
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
  }

  return JSON.stringify(content);
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

async function main(): Promise<void> {
  try {
    const server = await createServer();
    const port = process.env.OPENAI_AGENT_PORT ? Number(process.env.OPENAI_AGENT_PORT) : 5050;
    await server.listen({ port, host: "0.0.0.0" });
    console.log(`OpenAI-compatible API listening on http://localhost:${port}`);
  } catch (error) {
    console.error("Failed to start OpenAI-compatible server", error);
    process.exitCode = 1;
  }
}

void main();
