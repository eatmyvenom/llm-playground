import "reflect-metadata";
import { ValidationPipe } from "@nestjs/common";
import { NestFactory } from "@nestjs/core";
import { DocumentBuilder, SwaggerModule } from "@nestjs/swagger";
import { AppModule } from "./app.module.js";

async function bootstrap(): Promise<void> {
  const app = await NestFactory.create(AppModule);
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));

  const config = new DocumentBuilder()
    .setTitle("LLM Tools Server")
    .setDescription("OpenAPI + MCP endpoints for tooling integration")
    .setVersion("0.1.0")
    .build();

  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup("api", app, document);

  const port = process.env.PORT ? Number(process.env.PORT) : 4000;
  await app.listen(port);
  console.log(`Tools server running on http://localhost:${port}`);
}

void bootstrap();
