import { Module } from "@nestjs/common";
import { AppController } from "./app.controller.js";
import { AppService } from "./app.service.js";
import { McpController } from "./mcp/mcp.controller.js";
import { McpService } from "./mcp/mcp.service.js";
import { SampleDatabaseTool } from "./tools/sampleTool.js";
import { SmartWebSearchTool } from "./tools/smartSearchTool.js";

@Module({
  controllers: [AppController, McpController],
  providers: [AppService, McpService, SampleDatabaseTool, SmartWebSearchTool],
})
export class AppModule {}
