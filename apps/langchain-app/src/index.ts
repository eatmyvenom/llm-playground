import { HumanMessage } from "@langchain/core/messages";
import { buildReactAgent } from "./agents/reactAgent.js";
import { buildPlanningAgent } from "./agents/planningExecutorAgent.js";
import { logger } from "./logger.js";

async function main(): Promise<void> {
  try {
    const planningAgent = buildPlanningAgent();
    logger.info("Planning agent output:");
    process.stdout.write("\n");
    const planningStream = planningAgent.streamTask(
      "Create a long-horizon approach for improving the developer onboarding experience for this monorepo.",
    );
    for await (const chunk of planningStream) {
      process.stdout.write(chunk);
    }

    process.stdout.write("\n\n");
    logger.info("ReAct agent output:");
    process.stdout.write("\n");

    const reactAgent = await buildReactAgent();
    const response = await reactAgent.invoke({
      messages: [new HumanMessage("Retrieve the status of record alpha and summarize it.")],
    });

    const finalMessage = response.messages.at(-1);
    if (finalMessage) {
      const finalContent =
        typeof finalMessage.content === "string" ? finalMessage.content : JSON.stringify(finalMessage.content, null, 2);
      logger.info(finalContent);
    } else {
      logger.info(response);
    }
  } catch (error) {
    logger.error("Failed to run LangChain agent", error);
    process.exitCode = 1;
  }
}

void main();
