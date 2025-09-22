import { HumanMessage } from "@langchain/core/messages";
import { buildReactAgent } from "./agents/reactAgent.js";

async function main(): Promise<void> {
  try {
    const agent = await buildReactAgent();
    const response = await agent.invoke({
      messages: [new HumanMessage("Retrieve the status of record alpha and summarize it.")],
    });

    const finalMessage = response.messages.at(-1);
    if (finalMessage) {
      const finalContent =
        typeof finalMessage.content === "string" ? finalMessage.content : JSON.stringify(finalMessage.content, null, 2);
      console.log("ReAct agent response:\n", finalContent);
    } else {
      console.log("ReAct agent response:\n", response);
    }
  } catch (error) {
    console.error("Failed to run LangChain agent:", error);
    process.exitCode = 1;
  }
}

void main();
