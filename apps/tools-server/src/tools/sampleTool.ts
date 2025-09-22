import { BadRequestException, Injectable } from "@nestjs/common";
import {
  type McpToolCallResult,
  ReflectiveMcpTool,
  toolField,
} from "./baseTool.js";

interface MockRow {
  id: string;
  title: string;
  details: string;
}

const MOCK_DATA: Array<MockRow> = [
  {
    id: "alpha",
    title: "LLM usage summary",
    details: "Latest usage metrics for LLM-enabled workflows.",
  },
  {
    id: "beta",
    title: "Agent latency audit",
    details: "Latency analysis for agent execution pipeline.",
  },
];

class SampleDatabaseInput {
  @toolField.string({ description: "Record identifier (e.g. alpha)" })
  id!: string;
}

@Injectable()
export class SampleDatabaseTool extends ReflectiveMcpTool<SampleDatabaseInput> {
  readonly name = "mock_db_lookup";
  readonly description =
    "Lookup a mock record by id and return summary details.";

  protected readonly inputConstructor = SampleDatabaseInput;

  async call(args: Record<string, unknown>): Promise<McpToolCallResult> {
    const input = this.parseArgs(args);
    const id = input.id.trim();
    if (!id) {
      throw new BadRequestException("Invalid payload: id is required");
    }

    const record = MOCK_DATA.find((entry) => entry.id === id.toLowerCase());
    if (!record) {
      return { content: `No record found for id: ${id}` };
    }

    return {
      content: `${record.title}: ${record.details}`,
    };
  }
}
