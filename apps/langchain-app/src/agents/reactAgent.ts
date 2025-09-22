import "dotenv/config";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { createAgent } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

type MockRecord = {
  id: string;
  summary: string;
  status: "open" | "closed";
};

const mockRecords: Array<MockRecord> = [
  {
    id: "alpha",
    summary: "Investigate latency spikes in EU region",
    status: "open",
  },
  {
    id: "beta",
    summary: "Prototype LangChain ReAct agent for support triage",
    status: "closed",
  },
  { id: "gamma", summary: "Draft quarterly LLM ops report", status: "open" },
];

const queryRecordsTool = new DynamicStructuredTool({
  name: "query_mock_records",
  description: "Look up summary and status for a mock record id.",
  schema: z.object({
    id: z.string().describe("Record identifier (e.g. alpha)"),
  }),
  func: async ({ id }: { id: string }) => {
    const record = mockRecords.find((entry) => entry.id === id.toLowerCase());
    if (!record) {
      return `No record found for id: ${id}`;
    }
    return `Record ${record.id}: ${record.summary} (status: ${record.status})`;
  },
});

const toolsServerUrl = process.env.TOOLS_SERVER_URL ?? "http://localhost:4000";

const smartWebSearchTool = new DynamicStructuredTool({
  name: "web_search",
  description:
    "Perform a multi-source web search via the tools server. Returns structured JSON with query breakdown and top results.",
  schema: z.object({
    query: z.string().min(3).max(512).describe("Search query to execute"),
    allowSplit: z
      .boolean()
      .optional()
      .describe("Allow heuristic query splitting (default true)"),
    engine: z
      .enum(["brave", "tavily", "exa", "firecrawl"])
      .optional()
      .describe("Preferred engine override"),
    maxResults: z
      .number()
      .int()
      .min(1)
      .max(10)
      .optional()
      .describe("Limit results per query"),
  }),
  func: async (input) => {
    const response = await fetch(`${toolsServerUrl}/mcp/tools/call`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jsonrpc: "2.0",
        method: "tools.call",
        params: { name: "web_search", arguments: input },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `tools-server error (${response.status}): ${errorText || response.statusText}`,
      );
    }

    const payload = (await response.json()) as {
      result?: { content?: string };
    };

    if (!payload.result) {
      throw new Error("tools-server returned an empty result for web_search");
    }

    return payload.result.content ?? JSON.stringify(payload.result);
  },
});

export async function buildReactAgent() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to run the ReAct agent.");
  }

  const llm = new ChatOpenAI({
    model: "gpt-4.1-mini",
    temperature: 0,
    apiKey,
  });

  // Tools array is the initial integration point; MCP tool registry can be mapped here.
  const tools = [smartWebSearchTool, queryRecordsTool];
  return createAgent({ llm, tools });
}
