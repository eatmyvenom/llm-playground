import { Injectable, NotFoundException } from "@nestjs/common";
import { SampleDatabaseTool } from "../tools/sampleTool.js";
import { type McpTool, type McpToolDefinition } from "../tools/baseTool.js";
import { SmartWebSearchTool } from "../tools/smartSearchTool.js";

export interface McpListResponse {
  tools: Array<McpToolDefinition>;
}

export interface McpCallPayload {
  name: string;
  arguments: Record<string, unknown>;
}

@Injectable()
export class McpService {
  private readonly toolRegistry: Map<string, McpTool>;

  constructor(
    private readonly sampleTool: SampleDatabaseTool,
    private readonly smartSearchTool: SmartWebSearchTool,
  ) {
    this.toolRegistry = new Map(
      [sampleTool, smartSearchTool].map((tool) => [tool.name, tool]),
    );
  }

  listTools(): McpListResponse {
    return {
      tools: Array.from(this.toolRegistry.values()).map((tool) =>
        tool.listDefinition(),
      ),
    };
  }

  async callTool(payload: McpCallPayload) {
    const tool = this.toolRegistry.get(payload.name);
    if (!tool) {
      throw new NotFoundException(`Tool not found: ${payload.name}`);
    }

    return tool.call(payload.arguments);
  }
}
