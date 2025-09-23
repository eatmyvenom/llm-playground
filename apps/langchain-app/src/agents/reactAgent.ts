import "dotenv/config";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";
import { z } from "zod";

const DEFAULT_SYSTEM_PROMPT =
  "You are a focused research copilot. Use the available web search tool to gather evidence, keep responses concise, and clearly cite the data you relied on.";

type ToolsCallPayload = {
  jsonrpc: "2.0";
  method: "tools.call";
  params: { name: string; arguments: Record<string, unknown> };
};

type ToolsCallResponse = {
  result?: { content?: unknown };
};

export type BuildReactAgentOptions = {
  openAIApiKey?: string;
  systemPrompt?: string;
  toolsServerUrl?: string;
};

const buildSmartWebSearchTool = (baseUrl: string) => {
  const toolsEndpoint = new URL("/mcp/tools/call", baseUrl).toString();

  return new DynamicStructuredTool({
    name: "web_search",
    description:
      "Perform a multi-source web search via the tools server. Use this to find fresh information across the public web.",
    schema: z.object({
      query: z.string().min(3).max(512).describe("Search query to execute"),
      allowSplit: z.boolean().optional().describe("Allow heuristic query splitting (default true)"),
      engine: z.enum(["brave", "tavily", "exa", "firecrawl"]).optional().describe("Preferred engine override"),
      maxResults: z.number().int().min(1).max(10).optional().describe("Limit results per query"),
    }),
    func: async (input) => {
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

      return typeof content === "string" ? content : JSON.stringify(content);
    },
  });
};

export async function buildReactAgent(options: BuildReactAgentOptions = {}) {
  const apiKey = options.openAIApiKey ?? process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to run the ReAct agent.");
  }

  const toolsServerUrl = options.toolsServerUrl ?? process.env.TOOLS_SERVER_URL ?? "http://localhost:4000";
  const smartWebSearchTool = buildSmartWebSearchTool(toolsServerUrl);

  const llm = new ChatOpenAI({
    model: "gpt-5-mini",
    temperature: 0,
    apiKey,
  });

  const systemPrompt = options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;

  return createAgent({
    llm,
    tools: [smartWebSearchTool],
    prompt: systemPrompt,
    name: "web-search-react-agent",
    description: "Performs multi-source web research via the MCP tools server.",
  });
}
