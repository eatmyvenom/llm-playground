import { randomUUID } from "node:crypto";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { createPreview, createToolLogger } from "./toolLogging.js";

type ToolsCallPayload = {
  jsonrpc: "2.0";
  method: "tools.call";
  params: { name: string; arguments: Record<string, unknown> };
};

type ToolsCallResponse = {
  result?: { content?: unknown };
};

export function createWebSearchTool(baseUrl: string): DynamicStructuredTool {
  const toolsEndpoint = new URL("/mcp/tools/call", baseUrl).toString();
  const toolLogger = createToolLogger("web_search");

  return new DynamicStructuredTool({
    name: "web_search",
    description: [
      "Perform a multi-source web search via the tools server. Use this to find fresh information across the public web.",
      "Input schema: { query: string; allowSplit?: boolean; engine?: 'brave'|'tavily'|'exa'|'firecrawl'; maxResults?: number }.",
    ].join(" "),
    schema: z.object({
      query: z.string().min(3).max(512).describe("Search query to execute"),
      allowSplit: z.boolean().optional().describe("Allow heuristic query splitting (default true)"),
      engine: z.enum(["brave", "tavily", "exa", "firecrawl"]).optional().describe("Preferred engine override"),
      maxResults: z.number().int().min(1).max(10).optional().describe("Limit results per query"),
    }),
    func: async (input) => {
      const invocationId = randomUUID();
      const startedAt = Date.now();
      toolLogger.info("Invocation started", {
        invocationId,
        queryPreview: createPreview(input.query, 120),
        engine: input.engine ?? "auto",
        allowSplit: input.allowSplit,
        maxResults: input.maxResults,
      });

      try {
        const payload: ToolsCallPayload = {
          jsonrpc: "2.0",
          method: "tools.call",
          params: { name: "web_search", arguments: input },
        };

        const response = await fetch(toolsEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`tools-server error (${response.status}): ${errorText || response.statusText}`);
        }

        const result = (await response.json()) as ToolsCallResponse;
        const content = result.result?.content;

        if (content === undefined || content === null) {
          throw new Error("tools-server returned an empty result for web_search");
        }

        const serialized = typeof content === "string" ? content : JSON.stringify(content);
        toolLogger.info("Invocation completed", {
          invocationId,
          durationMs: Date.now() - startedAt,
          responsePreview: createPreview(serialized, 200),
        });
        return serialized;
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
