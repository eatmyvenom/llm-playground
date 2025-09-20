import { Injectable, NotFoundException } from "@nestjs/common";
import { SampleDatabaseTool } from "../tools/sampleTool.js";

export interface McpListResponse {
  tools: ReturnType<SampleDatabaseTool["listDefinition"]>[];
}

export interface McpCallPayload {
  name: string;
  arguments: Record<string, unknown>;
}

@Injectable()
export class McpService {
  constructor(private readonly sampleTool: SampleDatabaseTool) {}

  listTools(): McpListResponse {
    return {
      tools: [this.sampleTool.listDefinition()]
    };
  }

  async callTool(payload: McpCallPayload) {
    if (payload.name !== this.sampleTool.name) {
      throw new NotFoundException(`Tool not found: ${payload.name}`);
    }

    return this.sampleTool.call(payload.arguments as { id: string });
  }
}
