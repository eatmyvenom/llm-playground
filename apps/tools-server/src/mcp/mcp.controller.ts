import { Body, Controller, Post } from "@nestjs/common";
import { ApiTags } from "@nestjs/swagger";
import { Type } from "class-transformer";
import { IsObject, IsOptional, IsString, ValidateNested } from "class-validator";
import { McpService } from "./mcp.service.js";

class McpListRequestDto {
  @IsString()
  jsonrpc!: string;

  @IsOptional()
  id?: string | number | null;
}

class McpCallParamsDto {
  @IsString()
  name!: string;

  @IsObject()
  arguments!: Record<string, unknown>;
}

class McpCallRequestDto extends McpListRequestDto {
  @IsString()
  method!: string;

  @ValidateNested()
  @Type(() => McpCallParamsDto)
  params!: McpCallParamsDto;
}

@ApiTags("mcp")
@Controller("mcp")
export class McpController {
  constructor(private readonly mcpService: McpService) {}

  @Post("tools/list")
  list(@Body() body: McpListRequestDto) {
    return {
      jsonrpc: body.jsonrpc ?? "2.0",
      result: this.mcpService.listTools()
    };
  }

  @Post("tools/call")
  async call(@Body() body: McpCallRequestDto) {
    const result = await this.mcpService.callTool(body.params);
    return {
      jsonrpc: body.jsonrpc ?? "2.0",
      result
    };
  }
}
