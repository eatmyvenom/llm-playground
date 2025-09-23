import "../env.js";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatOpenAI } from "@langchain/openai";
import type { AIMessageChunk } from "@langchain/core/messages";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { DynamicStructuredTool } from "@langchain/core/tools";
import { createWebSearchTool } from "../tools/webSearchTool.js";
import { createCodeExecutionTool, type CodeExecutionHandler } from "../tools/codeExecutionTool.js";
import { createSchema, ZodField, z } from "./zodDecorators.js";

type PlanStep = z.infer<typeof planStepSchema>;
type Plan = z.infer<typeof planSchema>;
type ExecutionDecision = z.infer<typeof executionDecisionSchema>;
type ToolRequest = z.infer<typeof toolRequestSchema>;
type ToolExecutionRecord = {
  toolId: string;
  input: Record<string, unknown>;
  output: string;
  success: boolean;
  error?: string;
  rationale?: string;
};
type ExecutionLogEntry = {
  step: PlanStep;
  status: ExecutionDecision["status"];
  output: string;
  notes?: string;
  planChanges?: ExecutionDecision["planAdjustments"];
  toolResults?: Array<ToolExecutionRecord>;
};
type StructuredExecutor = {
  invoke: (input: string) => Promise<unknown>;
};

export type OpenAIModelSpecifier = {
  provider: "openai";
  model: string;
  apiKey?: string;
};

export type ModelFactory = () => BaseChatModel;
export type ModelSpec = BaseChatModel | ModelFactory | OpenAIModelSpecifier;

export interface BuildPlanningAgentOptions {
  /** Optional default planner model specification. */
  plannerModel?: ModelSpec;
  /** Optional default executor model specification. */
  executorModel?: ModelSpec;
  /** Optional default responder model specification. */
  responderModel?: ModelSpec;
  /** Optional base URL for the MCP tools server. Defaults to TOOLS_SERVER_URL env or http://localhost:4000. */
  toolsServerUrl?: string;
  /** Optional handler used when the code_execution tool is invoked. */
  codeExecutionHandler?: CodeExecutionHandler;
}

export interface PlanningAgentRunOverrides {
  plannerModel?: ModelSpec;
  executorModel?: ModelSpec;
  responderModel?: ModelSpec;
  toolsServerUrl?: string;
  codeExecutionHandler?: CodeExecutionHandler;
}

export interface PlanningAgent {
  streamTask(input: string, overrides?: PlanningAgentRunOverrides): AsyncGenerator<string>;
}

class PlanStepModel {
  @ZodField(() =>
    z.string().min(1).describe("Short identifier for the step. Prefer snake_case (e.g. research_background)."),
  )
  id!: string;

  @ZodField(() => z.string().min(1).describe("Concrete action the agent will take during this step."))
  description!: string;

  @ZodField(() => z.string().min(1).describe("Why this step is needed to accomplish the task."))
  rationale!: string;

  @ZodField(() => z.string().min(1).describe("What success looks like for this step."))
  expectedOutcome!: string;
}

const planStepSchema = createSchema(PlanStepModel);

class PlanModel {
  @ZodField(() => z.string().min(1).describe("One paragraph summary of how the task will be solved."))
  overview!: string;

  @ZodField(() =>
    z.array(z.string().min(1)).min(1).describe("Key assumptions or unknowns to track while executing the plan."),
  )
  assumptions!: Array<string>;

  @ZodField(() => z.array(planStepSchema).min(2).describe("Ordered multi-step plan to accomplish the task."))
  steps!: Array<PlanStepModel>;
}

const planSchema = createSchema(PlanModel);

class ExecutionPlanAdjustmentsModel {
  @ZodField(() => z.string().min(1).describe("Why the plan needs to change.").optional())
  reason?: string;

  @ZodField(() =>
    z.array(z.string().min(1)).describe("Steps that should be removed because they are unnecessary now.").optional(),
  )
  removeStepIds?: Array<string>;

  @ZodField(() =>
    z
      .string()
      .describe(
        "Identifier of the step after which new steps should be inserted. Defaults to the current step if omitted.",
      )
      .optional(),
  )
  insertAfterStepId?: string;

  @ZodField(() => z.array(planStepSchema).describe("Additional steps to add to the plan in order.").optional())
  newSteps?: Array<PlanStepModel>;
}

const planAdjustmentsSchema = createSchema(ExecutionPlanAdjustmentsModel, {
  description: "Describe any changes to the plan discovered during execution.",
});

class ToolRequestModel {
  @ZodField(() => z.string().min(1).describe("Identifier of the tool to call (e.g. web_search)."))
  toolId!: string;

  @ZodField(() => z.record(z.unknown()).describe("Arguments to pass to the tool as a JSON object."))
  input!: Record<string, unknown>;

  @ZodField(() => z.string().describe("Why this tool request is needed.").optional())
  rationale?: string;
}

const toolRequestSchema = createSchema(ToolRequestModel, {
  description: "Declare external tool invocations required before finalizing the step.",
});

class ExecutionDecisionModel {
  @ZodField(() => z.string().min(1).describe("Identifier of the step being executed."))
  stepId!: string;

  @ZodField(() =>
    z
      .enum(["completed", "skipped", "blocked"])
      .describe("Did the step complete, get skipped as unnecessary, or is it blocked?"),
  )
  status!: "completed" | "skipped" | "blocked";

  @ZodField(() => z.string().min(1).describe("Detailed write-up of what happened while working on this step."))
  output!: string;

  @ZodField(() => z.string().describe("Optional follow-up notes or todos uncovered while executing.").optional())
  notes?: string;

  @ZodField(() =>
    planAdjustmentsSchema.optional().describe("Describe any changes to the plan discovered during execution."),
  )
  planAdjustments?: z.infer<typeof planAdjustmentsSchema>;

  @ZodField(() =>
    z
      .array(toolRequestSchema)
      .min(1)
      .describe(
        "List of tools that must be executed before the step can be completed. Leave empty when no tools are required.",
      )
      .optional(),
  )
  toolRequests?: Array<ToolRequestModel>;
}

const executionDecisionSchema = createSchema(ExecutionDecisionModel);

const defaultPlannerModelName = process.env.PLANNING_AGENT_PLANNER_MODEL ?? "gpt-4.1";
const defaultExecutorModelName = process.env.PLANNING_AGENT_EXECUTOR_MODEL ?? "gpt-4.1-mini";
const defaultResponderModelName = process.env.PLANNING_AGENT_RESPONDER_MODEL ?? "gpt-4.1-mini";
const MAX_EXECUTION_TOOL_ITERATIONS = 3;

export function buildPlanningAgent(options?: BuildPlanningAgentOptions): PlanningAgent {
  const defaultPlannerFactory = normalizeModelFactory(options?.plannerModel, () =>
    createOpenAIModel({
      provider: "openai",
      model: defaultPlannerModelName,
    }),
  );

  const defaultExecutorFactory = normalizeModelFactory(options?.executorModel, () =>
    createOpenAIModel({
      provider: "openai",
      model: defaultExecutorModelName,
    }),
  );

  const defaultResponderFactory = normalizeModelFactory(options?.responderModel, () =>
    createOpenAIModel({
      provider: "openai",
      model: defaultResponderModelName,
    }),
  );

  return {
    async *streamTask(input: string, overrides?: PlanningAgentRunOverrides): AsyncGenerator<string> {
      if (!input.trim()) {
        throw new Error("Planning agent requires a non-empty task description.");
      }

      const planner = normalizeModelFactory(overrides?.plannerModel, defaultPlannerFactory)();
      const executor = normalizeModelFactory(overrides?.executorModel, defaultExecutorFactory)();
      const responder = normalizeModelFactory(overrides?.responderModel, defaultResponderFactory)();

      const toolsServerUrl =
        overrides?.toolsServerUrl ?? options?.toolsServerUrl ?? process.env.TOOLS_SERVER_URL ?? "http://localhost:4000";
      const codeExecutionHandler = overrides?.codeExecutionHandler ?? options?.codeExecutionHandler;
      const planningTools = createPlanningTools({ toolsServerUrl, codeExecutionHandler });
      const toolRegistry = new Map(planningTools.map((tool) => [tool.name, tool]));

      const structuredPlanner = planner.withStructuredOutput(planSchema, {
        name: "long_horizon_plan",
      });

      yield `<thinking>\nPlanning task...\n`;
      const plan = (await structuredPlanner.invoke(
        `You are an elite project planner. Create a multi-step plan (at least two steps) to solve the user's task. ` +
          `Steps must be long-horizon friendly, fully ordered, and have clear success criteria. ` +
          `Focus on reasoning about dependencies, potential research, and execution feasibility.\n` +
          `Task: ${input}`,
      )) as Plan;
      yield formatPlan(plan);
      yield "</thinking>\n";

      const executionLog: Array<ExecutionLogEntry> = [];
      const structuredExecutor = executor.withStructuredOutput(executionDecisionSchema, {
        name: "plan_execution_update",
      });

      for (let index = 0; index < plan.steps.length; index += 1) {
        const currentStep = plan.steps[index];
        const { decision, toolHistory } = await executePlanStep({
          executor: structuredExecutor,
          task: input,
          plan,
          step: currentStep,
          priorLog: executionLog,
          tools: planningTools,
          toolRegistry,
        });

        applyPlanAdjustments(plan, decision.planAdjustments, currentStep.id);

        executionLog.push({
          step: currentStep,
          status: decision.status,
          output: decision.output,
          notes: decision.notes,
          planChanges: decision.planAdjustments,
          toolResults: toolHistory.length ? toolHistory : undefined,
        });

        yield formatExecutionThought({
          index,
          decision,
          currentStep,
          toolsUsed: toolHistory,
        });

        if (decision.status === "blocked") {
          break;
        }

        // Adjust index if the current step was removed (rare but possible) or if new steps were inserted earlier.
        index = Math.max(
          -1,
          plan.steps.findIndex((step: PlanStep) => step.id === currentStep.id),
        );
      }

      yield "<thinking>\nPreparing final response...\n</thinking>\n";

      const responseMessages = [
        new SystemMessage(
          "You are a project executor summarizing planning and execution logs. Provide a concise, actionable final response.",
        ),
        new HumanMessage({
          content: [
            {
              type: "text",
              text: `Original task: ${input}`,
            },
            {
              type: "text",
              text: `Plan overview: ${plan.overview}`,
            },
            {
              type: "text",
              text: `Assumptions: ${plan.assumptions.join(", ")}`,
            },
            {
              type: "text",
              text: `Execution log:\n${formatExecutionLog(executionLog)}`,
            },
          ],
        }),
      ];

      const stream = await responder.stream(responseMessages);
      for await (const chunk of stream) {
        const text = extractTextFromChunk(chunk);
        if (text) {
          yield text;
        }
      }
    },
  };
}

function normalizeModelFactory(spec: ModelSpec | undefined, fallback: ModelFactory): ModelFactory {
  if (!spec) {
    return fallback;
  }

  if (typeof spec === "function") {
    return spec;
  }

  if (isOpenAISpecifier(spec)) {
    return () => createOpenAIModel(spec);
  }

  return () => spec;
}

function isOpenAISpecifier(value: ModelSpec): value is OpenAIModelSpecifier {
  return typeof value === "object" && value !== null && "provider" in value && value.provider === "openai";
}

function createOpenAIModel(spec: OpenAIModelSpecifier): BaseChatModel {
  const apiKey = spec.apiKey ?? process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required to run the planning agent with OpenAI models.");
  }

  return new ChatOpenAI({
    apiKey,
    model: spec.model,
  });
}

function createPlanningTools(args: {
  toolsServerUrl: string;
  codeExecutionHandler?: CodeExecutionHandler;
}): Array<DynamicStructuredTool> {
  const tools: Array<DynamicStructuredTool> = [];
  tools.push(createWebSearchTool(args.toolsServerUrl));
  tools.push(createCodeExecutionTool(args.codeExecutionHandler));
  return tools;
}

async function executePlanStep(args: {
  executor: StructuredExecutor;
  task: string;
  plan: Plan;
  step: PlanStep;
  priorLog: Array<ExecutionLogEntry>;
  tools: Array<DynamicStructuredTool>;
  toolRegistry: Map<string, DynamicStructuredTool>;
}): Promise<{ decision: ExecutionDecision; toolHistory: Array<ToolExecutionRecord> }> {
  const { executor, task, plan, step, priorLog, tools, toolRegistry } = args;
  const toolHistory: Array<ToolExecutionRecord> = [];
  let finalDecision: ExecutionDecision | undefined;
  let lastDecision: ExecutionDecision | undefined;
  let toolLoops = 0;

  while (toolLoops <= MAX_EXECUTION_TOOL_ITERATIONS) {
    const executionPrompt = buildExecutionPrompt({
      task,
      plan,
      step,
      priorLog,
      tools,
      toolHistory,
    });

    const candidateDecision = (await executor.invoke(executionPrompt)) as ExecutionDecision;
    lastDecision = candidateDecision;
    const requests = candidateDecision.toolRequests ?? [];

    if (!requests.length) {
      finalDecision = candidateDecision;
      break;
    }

    if (toolLoops >= MAX_EXECUTION_TOOL_ITERATIONS) {
      finalDecision = {
        ...candidateDecision,
        notes: appendNote(
          candidateDecision.notes,
          "Tool request limit reached before completion. Summarize remaining follow-ups explicitly.",
        ),
        toolRequests: undefined,
      };
      break;
    }

    const results = await executeToolRequests(requests, toolRegistry);
    toolHistory.push(...results);
    toolLoops += 1;
  }

  if (!finalDecision) {
    if (!lastDecision) {
      throw new Error(`Executor failed to produce a decision for step ${step.id}.`);
    }
    finalDecision = lastDecision;
  }

  if (finalDecision.toolRequests) {
    // Ensure downstream consumers do not attempt to re-run the same tool requests.
    finalDecision = { ...finalDecision, toolRequests: undefined };
  }

  return { decision: finalDecision, toolHistory };
}

async function executeToolRequests(
  requests: Array<ToolRequest>,
  toolRegistry: Map<string, DynamicStructuredTool>,
): Promise<Array<ToolExecutionRecord>> {
  const results: Array<ToolExecutionRecord> = [];

  for (const request of requests) {
    const baseRecord: ToolExecutionRecord = {
      toolId: request.toolId,
      input: request.input,
      output: "",
      success: false,
      rationale: request.rationale,
    };

    const tool = toolRegistry.get(request.toolId);
    if (!tool) {
      const message = `Requested tool '${request.toolId}' is not registered.`;
      results.push({ ...baseRecord, output: message, error: message });
      continue;
    }

    try {
      const invocation = await tool.invoke(request.input as never);
      const output = typeof invocation === "string" ? invocation : JSON.stringify(invocation);
      results.push({ ...baseRecord, output, success: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown tool execution error.";
      results.push({ ...baseRecord, output: message, error: message });
    }
  }

  return results;
}

function appendNote(existing: string | undefined, addition: string): string {
  return existing ? `${existing} ${addition}` : addition;
}

function describeToolExecutions(executions: Array<ToolExecutionRecord>): string {
  return executions
    .map((record) => {
      const status = record.success ? "success" : "error";
      const rationale = record.rationale ? `Rationale: ${record.rationale}. ` : "";
      const inputPreview = JSON.stringify(record.input);
      const outputPreview = truncate(record.output, 400);
      return `${record.toolId} => ${status}. ${rationale}Input: ${inputPreview}. Output: ${outputPreview}`;
    })
    .join("\n");
}

function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}

function formatPlan(plan: Plan): string {
  const stepLines = plan.steps.map(
    (step: PlanStep, index: number) =>
      `${index + 1}. [${step.id}] ${step.description}\n   Rationale: ${step.rationale}\n   Success: ${step.expectedOutcome}`,
  );

  return [
    `Overview: ${plan.overview}`,
    `Assumptions:`,
    ...plan.assumptions.map((assumption: string) => `- ${assumption}`),
    `Steps:`,
    ...stepLines,
  ].join("\n");
}

function buildExecutionPrompt(args: {
  task: string;
  plan: Plan;
  step: PlanStep;
  priorLog: Array<ExecutionLogEntry>;
  tools: Array<DynamicStructuredTool>;
  toolHistory: Array<ToolExecutionRecord>;
}): string {
  const { task, plan, step, priorLog, tools, toolHistory } = args;
  const completed = priorLog
    .map(
      (entry) =>
        `Step ${entry.step.id} (${entry.step.description}) => status: ${entry.status}. Outcome: ${entry.output}${
          entry.notes ? ` Notes: ${entry.notes}` : ""
        }`,
    )
    .join("\n");

  const toolSummaries = tools.length
    ? tools.map((tool) => `- ${tool.name}: ${tool.description ?? "No description provided."}`).join("\n")
    : "- No external tools are currently registered.";

  const toolGuidance = [
    `If you need a tool, set status to 'blocked' and populate toolRequests with entries like { "toolId": "web_search", "input": { ... }, "rationale": "why this helps" }.`,
    "After tool results are provided, respond again with toolRequests omitted and a final status (completed/skipped/blocked).",
    `You may request tools up to ${MAX_EXECUTION_TOOL_ITERATIONS} time(s) per step before finalizing.`,
  ].join(" \n");

  const priorToolRuns = toolHistory.length
    ? `Tool results available:\n${describeToolExecutions(toolHistory)}`
    : "No tool results have been provided yet.";

  return [
    "You are executing a step from a previously generated plan.",
    "Follow the plan but adapt if information changes.",
    "You may suggest new steps or remove future steps when justified.",
    "Tools are available to help you gather information or perform execution. Use them deliberately.",
    `Available tools:\n${toolSummaries}`,
    toolGuidance,
    priorToolRuns,
    "Return JSON that matches the provided schema.",
    `Task: ${task}`,
    `Plan overview: ${plan.overview}`,
    `Current step (${step.id}): ${step.description}`,
    `Expected outcome: ${step.expectedOutcome}`,
    plan.assumptions.length ? `Tracked assumptions: ${plan.assumptions.join(", ")}` : "",
    completed ? `Prior progress:\n${completed}` : "No prior steps have been executed yet.",
  ]
    .filter(Boolean)
    .join("\n\n");
}

function applyPlanAdjustments(
  plan: Plan,
  adjustments: ExecutionDecision["planAdjustments"],
  currentStepId: string,
): void {
  if (!adjustments) {
    return;
  }

  if (adjustments.removeStepIds?.length) {
    const removals = new Set(adjustments.removeStepIds);
    plan.steps = plan.steps.filter((step: PlanStep) => !removals.has(step.id));
  }

  if (adjustments.newSteps?.length) {
    const insertionAnchor = adjustments.insertAfterStepId;
    const anchorIndex = insertionAnchor
      ? plan.steps.findIndex((step: PlanStep) => step.id === insertionAnchor)
      : plan.steps.findIndex((step: PlanStep) => step.id === currentStepId);
    const safeIndex = anchorIndex >= 0 ? anchorIndex + 1 : plan.steps.length;

    const existingIds = new Set(plan.steps.map((step: PlanStep) => step.id));

    const normalizedSteps = adjustments.newSteps.map((step: PlanStep) => {
      let identifier = step.id;
      let counter = 1;
      while (existingIds.has(identifier)) {
        identifier = `${step.id}_${counter}`;
        counter += 1;
      }
      existingIds.add(identifier);
      return { ...step, id: identifier } as PlanStep;
    });

    plan.steps.splice(safeIndex, 0, ...normalizedSteps);
  }
}

function formatExecutionThought(args: {
  index: number;
  decision: ExecutionDecision;
  currentStep: PlanStep;
  toolsUsed?: Array<ToolExecutionRecord>;
}): string {
  const { index, decision, currentStep, toolsUsed } = args;
  const lines = [
    `<thinking>`,
    `Step ${index + 1} (${currentStep.id})`,
    `Executing: ${currentStep.description}`,
    `Status: ${decision.status}`,
    `Details: ${decision.output}`,
  ];

  if (toolsUsed && toolsUsed.length) {
    lines.push(`Tool usage:\n${describeToolExecutions(toolsUsed)}`);
  }

  if (decision.notes) {
    lines.push(`Notes: ${decision.notes}`);
  }

  if (decision.planAdjustments) {
    lines.push(`Plan adjustments: ${describeAdjustments(decision.planAdjustments)}`);
  }

  lines.push(`</thinking>\n`);
  return lines.join("\n");
}

function describeAdjustments(adjustments: NonNullable<ExecutionDecision["planAdjustments"]>): string {
  const parts: Array<string> = [];
  if (adjustments.reason) {
    parts.push(`Reason: ${adjustments.reason}`);
  }
  if (adjustments.removeStepIds?.length) {
    parts.push(`Remove: ${adjustments.removeStepIds.join(", ")}`);
  }
  if (adjustments.newSteps?.length) {
    const additions = adjustments.newSteps.map((step: PlanStep) => `[${step.id}] ${step.description}`).join("; ");
    parts.push(`Add: ${additions}`);
  }
  return parts.join(" | ");
}

function formatExecutionLog(entries: Array<ExecutionLogEntry>): string {
  return entries
    .map((entry) => {
      const lines: Array<string> = [];
      lines.push(`- ${entry.step.id} (${entry.step.description}) => ${entry.status}. Outcome: ${entry.output}`);
      if (entry.notes) {
        lines.push(`  Notes: ${entry.notes}`);
      }
      if (entry.planChanges) {
        lines.push(`  Plan changes: ${describeAdjustments(entry.planChanges)}`);
      }
      if (entry.toolResults?.length) {
        const toolDetails = describeToolExecutions(entry.toolResults)
          .split("\n")
          .map((detail) => `    ${detail}`)
          .join("\n");
        lines.push(`  Tools:\n${toolDetails}`);
      }
      return lines.join("\n");
    })
    .join("\n");
}

function extractTextFromChunk(chunk: AIMessageChunk): string {
  const { content } = chunk;
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item.type === "text") {
          return item.text;
        }
        return "";
      })
      .join("");
  }

  return "";
}
