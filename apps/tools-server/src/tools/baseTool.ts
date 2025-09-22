import "reflect-metadata";
import { BadRequestException } from "@nestjs/common";

export type McpToolDefinition = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
};

export type McpToolCallResult = {
  content: string;
};

export interface McpTool {
  readonly name: string;
  readonly description: string;
  listDefinition(): McpToolDefinition;
  call(args: Record<string, unknown>): Promise<McpToolCallResult>;
}

type ToolFieldType = "string" | "boolean" | "integer" | "number";

export type ToolFieldConfig = {
  description?: string;
  enum?: Array<string | number>;
  default?: unknown;
  required?: boolean;
  minimum?: number;
  maximum?: number;
  format?: string;
  examples?: Array<unknown>;
};

export type ToolFieldOptions = ToolFieldConfig & {
  type: ToolFieldType;
};

type ToolFieldDescriptor = {
  name: string;
  options: ToolFieldOptions;
};

type ToolInputConstructor<T> = new () => T;

type FieldOverrideMap = Record<string, Partial<ToolFieldOptions>>;

const TOOL_FIELDS_METADATA_KEY = Symbol("mcp:tool:fields");

function defineField(
  target: object,
  propertyKey: string | symbol,
  options: ToolFieldOptions,
): void {
  const constructor = target.constructor;
  const existing: Array<ToolFieldDescriptor> =
    Reflect.getOwnMetadata(TOOL_FIELDS_METADATA_KEY, constructor) ?? [];
  const filtered = existing.filter(
    (descriptor) => descriptor.name !== propertyKey,
  );
  const entry: ToolFieldDescriptor = {
    name: String(propertyKey),
    options: { ...options },
  };
  Reflect.defineMetadata(
    TOOL_FIELDS_METADATA_KEY,
    [...filtered, entry],
    constructor,
  );
}

export function toolFieldDecorator(
  options: ToolFieldOptions,
): PropertyDecorator {
  return (target, propertyKey) => {
    defineField(target, propertyKey, options);
  };
}

function createTypedDecorator(
  type: ToolFieldType,
  config: ToolFieldConfig = {},
): PropertyDecorator {
  return toolFieldDecorator({ ...config, type });
}

export const toolField = Object.freeze({
  string: (config: ToolFieldConfig = {}): PropertyDecorator =>
    createTypedDecorator("string", config),
  boolean: (config: ToolFieldConfig = {}): PropertyDecorator =>
    createTypedDecorator("boolean", config),
  integer: (config: ToolFieldConfig = {}): PropertyDecorator =>
    createTypedDecorator("integer", config),
  number: (config: ToolFieldConfig = {}): PropertyDecorator =>
    createTypedDecorator("number", config),
});

function collectFieldMetadata(
  constructor: ToolInputConstructor<object>,
): Array<ToolFieldDescriptor> {
  const parentPrototype = Object.getPrototypeOf(constructor.prototype);
  const parentConstructor =
    parentPrototype && parentPrototype !== Object.prototype
      ? parentPrototype.constructor
      : undefined;

  const parentFields = parentConstructor
    ? collectFieldMetadata(parentConstructor as ToolInputConstructor<object>)
    : [];

  const ownFields: Array<ToolFieldDescriptor> =
    Reflect.getOwnMetadata(TOOL_FIELDS_METADATA_KEY, constructor) ?? [];

  const merged = [...parentFields];
  for (const descriptor of ownFields) {
    const index = merged.findIndex((entry) => entry.name === descriptor.name);
    if (index >= 0) {
      merged.splice(index, 1, descriptor);
    } else {
      merged.push(descriptor);
    }
  }
  return merged;
}

function toJsonSchema(options: ToolFieldOptions): Record<string, unknown> {
  const schema: Record<string, unknown> = {
    type: options.type,
  };
  if (options.description) {
    schema.description = options.description;
  }
  if (options.enum) {
    schema.enum = options.enum;
  }
  if (Object.hasOwn(options, "default")) {
    schema.default = options.default;
  }
  if (Object.hasOwn(options, "minimum")) {
    schema.minimum = options.minimum;
  }
  if (Object.hasOwn(options, "maximum")) {
    schema.maximum = options.maximum;
  }
  if (options.format) {
    schema.format = options.format;
  }
  if (options.examples) {
    schema.examples = options.examples;
  }
  return schema;
}

function ensureType(
  name: string,
  value: unknown,
  options: ToolFieldOptions,
): void {
  switch (options.type) {
    case "string":
      if (typeof value !== "string") {
        throw new BadRequestException(
          `Invalid payload: ${name} must be a string`,
        );
      }
      break;
    case "boolean":
      if (typeof value !== "boolean") {
        throw new BadRequestException(
          `Invalid payload: ${name} must be a boolean`,
        );
      }
      break;
    case "integer":
      if (typeof value !== "number" || !Number.isInteger(value)) {
        throw new BadRequestException(
          `Invalid payload: ${name} must be an integer`,
        );
      }
      break;
    case "number":
      if (typeof value !== "number" || Number.isNaN(value)) {
        throw new BadRequestException(
          `Invalid payload: ${name} must be a number`,
        );
      }
      break;
    default:
      throw new BadRequestException(`Unsupported field type for ${name}`);
  }

  if (options.enum && !options.enum.includes(value as never)) {
    throw new BadRequestException(
      `Invalid payload: ${name} must be one of ${options.enum.join(", ")}`,
    );
  }

  if (typeof value === "number") {
    if (
      Object.hasOwn(options, "minimum") &&
      value < (options.minimum as number)
    ) {
      throw new BadRequestException(
        `Invalid payload: ${name} must be >= ${String(options.minimum)}`,
      );
    }
    if (
      Object.hasOwn(options, "maximum") &&
      value > (options.maximum as number)
    ) {
      throw new BadRequestException(
        `Invalid payload: ${name} must be <= ${String(options.maximum)}`,
      );
    }
  }
}

export abstract class ReflectiveMcpTool<
  SchemaInput extends object,
  ParsedInput extends object = SchemaInput,
> implements McpTool
{
  abstract readonly name: string;
  abstract readonly description: string;

  protected abstract readonly inputConstructor: ToolInputConstructor<SchemaInput>;

  abstract call(args: Record<string, unknown>): Promise<McpToolCallResult>;

  protected fieldOverrides(): FieldOverrideMap {
    return {};
  }

  protected transformInput(input: SchemaInput): ParsedInput {
    return input as unknown as ParsedInput;
  }

  listDefinition(): McpToolDefinition {
    const fieldMap = this.resolveFields();
    const properties: Record<string, unknown> = {};
    const required: Array<string> = [];

    for (const { name, options } of fieldMap) {
      properties[name] = toJsonSchema(options);
      if (options.required !== false) {
        required.push(name);
      }
    }

    const schema: Record<string, unknown> = {
      type: "object",
      properties,
      additionalProperties: false,
    };

    if (required.length) {
      schema.required = required;
    }

    return {
      name: this.name,
      description: this.description,
      inputSchema: schema,
    };
  }

  protected parseArgs(args: Record<string, unknown>): ParsedInput {
    if (!args || typeof args !== "object" || Array.isArray(args)) {
      throw new BadRequestException(
        "Invalid payload: arguments must be an object",
      );
    }

    const entries = this.resolveFields();
    const knownKeys = new Set(entries.map((entry) => entry.name));

    for (const key of Object.keys(args)) {
      if (!knownKeys.has(key)) {
        throw new BadRequestException(
          `Invalid payload: unexpected field ${key}`,
        );
      }
    }

    const normalized: Record<string, unknown> = {};

    for (const { name, options } of entries) {
      const hasDefault = Object.hasOwn(options, "default");
      const value = (args as Record<string, unknown>)[name];

      if (value === undefined || value === null) {
        if (hasDefault) {
          normalized[name] = options.default;
          continue;
        }
        if (options.required === false) {
          continue;
        }
        throw new BadRequestException(`Invalid payload: ${name} is required`);
      }

      ensureType(name, value, options);
      normalized[name] = value;
    }

    return this.transformInput(normalized as unknown as SchemaInput);
  }

  protected resolveFields(): Array<ToolFieldDescriptor> {
    const descriptors = collectFieldMetadata(this.inputConstructor);
    const overrides = this.fieldOverrides();
    if (!Object.keys(overrides).length) {
      return descriptors;
    }

    return descriptors.map((descriptor) => {
      const override = overrides[descriptor.name];
      if (!override) {
        return descriptor;
      }
      return {
        name: descriptor.name,
        options: { ...descriptor.options, ...override },
      };
    });
  }
}
