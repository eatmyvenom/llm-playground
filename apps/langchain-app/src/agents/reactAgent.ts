import "../env.js";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";
import { createWebSearchTool } from "../tools/webSearchTool.js";

const DEFAULT_SYSTEM_PROMPT =
  "You are a focused research copilot. Use the available web search tool to gather evidence, keep responses concise, and clearly cite the data you relied on.";

export type BuildReactAgentOptions = {
  openAIApiKey?: string;
  systemPrompt?: string;
  toolsServerUrl?: string;
};

export async function buildReactAgent(options: BuildReactAgentOptions = {}) {
  const apiKey = options.openAIApiKey ?? process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to run the ReAct agent.");
  }

  const toolsServerUrl = options.toolsServerUrl ?? process.env.TOOLS_SERVER_URL ?? "http://localhost:4000";
  const smartWebSearchTool = createWebSearchTool(toolsServerUrl);

  const llm = new ChatOpenAI({
    model: "gpt-5-mini",
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
