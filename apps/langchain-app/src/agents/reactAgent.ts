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
  const tools = [queryRecordsTool];
  return createAgent({ llm, tools });
}
