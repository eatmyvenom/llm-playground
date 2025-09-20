import { BadRequestException } from "@nestjs/common";

interface MockRow {
  id: string;
  title: string;
  details: string;
}

type ToolCallInput = {
  id: string;
};

const MOCK_DATA: MockRow[] = [
  { id: "alpha", title: "LLM usage summary", details: "Latest usage metrics for LLM-enabled workflows." },
  { id: "beta", title: "Agent latency audit", details: "Latency analysis for agent execution pipeline." }
];

export type McpToolDefinition = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
};

export type McpToolCallResult = {
  content: string;
};

export class SampleDatabaseTool {
  readonly name = "mock_db_lookup";
  readonly description = "Lookup a mock record by id and return summary details.";

  listDefinition(): McpToolDefinition {
    return {
      name: this.name,
      description: this.description,
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string", description: "Record identifier (e.g. alpha)" }
        },
        required: ["id"],
        additionalProperties: false
      }
    };
  }

  async call(args: ToolCallInput): Promise<McpToolCallResult> {
    if (!args || typeof args.id !== "string") {
      throw new BadRequestException("Invalid payload: id is required");
    }

    const record = MOCK_DATA.find((entry) => entry.id === args.id.toLowerCase());
    if (!record) {
      return { content: `No record found for id: ${args.id}` };
    }

    return {
      content: `${record.title}: ${record.details}`
    };
  }
}
