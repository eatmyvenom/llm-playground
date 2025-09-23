import { randomUUID } from "node:crypto";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { createPreview, createToolLogger } from "./toolLogging.js";

export interface CodeExecutionHandlerInput {
  language: string;
  snippet: string;
  context?: string;
  expectedOutput?: string;
}

export type CodeExecutionHandler = (input: CodeExecutionHandlerInput) => Promise<string> | string;

export function createCodeExecutionTool(handler?: CodeExecutionHandler): DynamicStructuredTool {
  const toolLogger = createToolLogger("code_execution");

  return new DynamicStructuredTool({
    name: "code_execution",
    description: [
      "Run code or shell snippets in a controlled environment when analysis requires concrete results.",
      "Input schema: { language: string; snippet: string; context?: string; expectedOutput?: string }.",
    ].join(" "),
    schema: z.object({
      language: z.string().min(1).max(64).describe("Language or runtime for the snippet (e.g. python, node, bash)."),
      snippet: z.string().min(1).describe("Executable code or command to run."),
      context: z
        .string()
        .max(2000)
        .optional()
        .describe("Additional notes or setup requirements for running the snippet."),
      expectedOutput: z
        .string()
        .max(2000)
        .optional()
        .describe("What the code is expected to produce. Helps validate execution results."),
    }),
    func: async (input) => {
      const invocationId = randomUUID();
      const startedAt = Date.now();
      toolLogger.info("Invocation started", {
        invocationId,
        language: input.language,
        snippetPreview: createPreview(input.snippet, 120),
        hasHandler: Boolean(handler),
      });

      try {
        if (!handler) {
          const fallback = [
            "Code execution environment is not configured yet.",
            "Provide the snippet to a human operator or run it locally as needed.",
            `Language: ${input.language}`,
            `Snippet:\n${input.snippet}`,
            input.context ? `Context: ${input.context}` : null,
            input.expectedOutput ? `Expected output: ${input.expectedOutput}` : null,
          ]
            .filter(Boolean)
            .join("\n");

          const durationMs = Date.now() - startedAt;
          toolLogger.warn("No handler configured", {
            invocationId,
            durationMs,
          });
          toolLogger.info("Invocation completed", {
            invocationId,
            durationMs,
            responsePreview: createPreview(fallback, 200),
          });
          return fallback;
        }

        const result = await handler(input);
        const output = typeof result === "string" ? result : String(result);
        toolLogger.info("Invocation completed", {
          invocationId,
          durationMs: Date.now() - startedAt,
          responsePreview: createPreview(output, 200),
        });
        return output;
      } catch (error) {
        toolLogger.error(
          "Invocation failed",
          {
            invocationId,
            durationMs: Date.now() - startedAt,
          },
          error,
        );
        throw error;
      }
    },
  });
}
