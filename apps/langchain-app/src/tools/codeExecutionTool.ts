import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";

export interface CodeExecutionHandlerInput {
  language: string;
  snippet: string;
  context?: string;
  expectedOutput?: string;
}

export type CodeExecutionHandler = (input: CodeExecutionHandlerInput) => Promise<string> | string;

export function createCodeExecutionTool(handler?: CodeExecutionHandler): DynamicStructuredTool {
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
      if (!handler) {
        return [
          "Code execution environment is not configured yet.",
          "Provide the snippet to a human operator or run it locally as needed.",
          `Language: ${input.language}`,
          `Snippet:\n${input.snippet}`,
          input.context ? `Context: ${input.context}` : null,
          input.expectedOutput ? `Expected output: ${input.expectedOutput}` : null,
        ]
          .filter(Boolean)
          .join("\n");
      }

      const result = await handler(input);
      return result;
    },
  });
}
