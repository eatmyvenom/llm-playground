import { createLogger, installProcessHandlers } from "@llm/logger";

const logger = createLogger({ name: "langchain-app" });
installProcessHandlers(logger);

export { logger };
