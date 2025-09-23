import { createLogger, installProcessHandlers } from "@llm/logger";

const logger = createLogger({ name: "tools-server" });
installProcessHandlers(logger);

export { logger };
