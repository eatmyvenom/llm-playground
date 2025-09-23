import "dotenv/config";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatOpenAI } from "@langchain/openai";
import type { AIMessageChunk } from "@langchain/core/messages";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { createSchema, ZodField, z } from "./zodDecorators.js";

type PlanStep = z.infer<typeof planStepSchema>;
type Plan = z.infer<typeof planSchema>;
type ExecutionDecision = z.infer<typeof executionDecisionSchema>;
type ExecutionLogEntry = {
  step: PlanStep;
  status: ExecutionDecision["status"];
  output: string;
  notes?: string;
  planChanges?: ExecutionDecision["planAdjustments"];
};

export type OpenAIModelSpecifier = {
  provider: "openai";
  model: string;
  temperature?: number;
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
}

export interface PlanningAgentRunOverrides {
  plannerModel?: ModelSpec;
  executorModel?: ModelSpec;
  responderModel?: ModelSpec;
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
}

const executionDecisionSchema = createSchema(ExecutionDecisionModel);

const defaultPlannerModelName = process.env.PLANNING_AGENT_PLANNER_MODEL ?? "gpt-4.1";
const defaultExecutorModelName = process.env.PLANNING_AGENT_EXECUTOR_MODEL ?? "gpt-4.1-mini";
const defaultResponderModelName = process.env.PLANNING_AGENT_RESPONDER_MODEL ?? "gpt-4.1-mini";

export function buildPlanningAgent(options?: BuildPlanningAgentOptions): PlanningAgent {
  const defaultPlannerFactory = normalizeModelFactory(options?.plannerModel, () =>
    createOpenAIModel({
      provider: "openai",
      model: defaultPlannerModelName,
      temperature: 0.1,
    }),
  );

  const defaultExecutorFactory = normalizeModelFactory(options?.executorModel, () =>
    createOpenAIModel({
      provider: "openai",
      model: defaultExecutorModelName,
      temperature: 0,
    }),
  );

  const defaultResponderFactory = normalizeModelFactory(options?.responderModel, () =>
    createOpenAIModel({
      provider: "openai",
      model: defaultResponderModelName,
      temperature: 0.2,
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
        const executionPrompt = buildExecutionPrompt({
          task: input,
          plan,
          step: currentStep,
          priorLog: executionLog,
        });

        const decision = (await structuredExecutor.invoke(executionPrompt)) as ExecutionDecision;

        applyPlanAdjustments(plan, decision.planAdjustments, currentStep.id);

        executionLog.push({
          step: currentStep,
          status: decision.status,
          output: decision.output,
          notes: decision.notes,
          planChanges: decision.planAdjustments,
        });

        yield formatExecutionThought({
          index,
          decision,
          currentStep,
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
    temperature: spec.temperature ?? 0,
  });
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
}): string {
  const { task, plan, step, priorLog } = args;
  const completed = priorLog
    .map(
      (entry) =>
        `Step ${entry.step.id} (${entry.step.description}) => status: ${entry.status}. Outcome: ${entry.output}${
          entry.notes ? ` Notes: ${entry.notes}` : ""
        }`,
    )
    .join("\n");

  return [
    "You are executing a step from a previously generated plan.",
    "Follow the plan but adapt if information changes.",
    "You may suggest new steps or remove future steps when justified.",
    "Reserve space for tool usage (e.g. future code execution) by clearly describing what needs to run.",
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

function formatExecutionThought(args: { index: number; decision: ExecutionDecision; currentStep: PlanStep }): string {
  const { index, decision, currentStep } = args;
  const lines = [
    `<thinking>`,
    `Step ${index + 1} (${currentStep.id})`,
    `Executing: ${currentStep.description}`,
    `Status: ${decision.status}`,
    `Details: ${decision.output}`,
  ];

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
      const base = `- ${entry.step.id} (${entry.step.description}) => ${entry.status}. Outcome: ${entry.output}`;
      const notes = entry.notes ? ` Notes: ${entry.notes}` : "";
      const adjustments = entry.planChanges ? ` Plan changes: ${describeAdjustments(entry.planChanges)}` : "";
      return `${base}${notes}${adjustments}`;
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
