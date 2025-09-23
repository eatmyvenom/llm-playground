import { createRequire } from "node:module";
import { z, type ZodTypeAny } from "zod";

/**
 * Runtime adapter that prefers the real `@anatine/zod-decorators` package when it is
 * available but gracefully falls back to a tiny local implementation during builds
 * where the dependency has not been installed yet.
 */

type SchemaFactory<T extends ZodTypeAny = ZodTypeAny> = () => T;

type DecoratorModule = {
  ZodField: (factory: SchemaFactory) => PropertyDecorator;
  createSchema: <T extends new () => unknown>(
    ctor: T,
    options?: { description?: string },
  ) => z.ZodObject<Record<string, ZodTypeAny>>;
  z: typeof z;
};

const require = createRequire(import.meta.url);

function tryLoadZodDecorators(): DecoratorModule | null {
  try {
    return require("@anatine/zod-decorators") as DecoratorModule;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "MODULE_NOT_FOUND") {
      console.warn("Failed to load @anatine/zod-decorators:", error);
    }
    return null;
  }
}

function createFallbackModule(): DecoratorModule {
  const fieldRegistry = new WeakMap<Function, Array<{ key: string; factory: SchemaFactory }>>();

  const getOrCreateFieldList = (target: Function) => {
    const existing = fieldRegistry.get(target);
    if (existing) {
      return existing;
    }
    const list: Array<{ key: string; factory: SchemaFactory }> = [];
    fieldRegistry.set(target, list);
    return list;
  };

  const ZodField =
    (factory: SchemaFactory): PropertyDecorator =>
    (prototype, propertyKey) => {
      const ctor = prototype.constructor as Function;
      const fields = getOrCreateFieldList(ctor);
      const key = typeof propertyKey === "string" ? propertyKey : propertyKey.toString();
      const existingIndex = fields.findIndex((field) => field.key === key);
      if (existingIndex >= 0) {
        fields.splice(existingIndex, 1);
      }
      fields.push({
        key,
        factory,
      });
    };

  const createSchema = <T extends new () => unknown>(ctor: T, options?: { description?: string }) => {
    const fields = fieldRegistry.get(ctor) ?? [];
    if (fields.length === 0) {
      throw new Error(`No decorated fields found for ${ctor.name}. Ensure properties are annotated with @ZodField().`);
    }

    const shape = fields.reduce<Record<string, ZodTypeAny>>((acc, field) => {
      acc[field.key] = field.factory();
      return acc;
    }, {});

    let schema = z.object(shape);
    if (options?.description) {
      schema = schema.describe(options.description);
    }

    return schema;
  };

  return {
    ZodField,
    createSchema,
    z,
  };
}

const loadedModule = tryLoadZodDecorators();
const { ZodField, createSchema } = loadedModule ?? createFallbackModule();

export { createSchema, ZodField };
export { z };
export type { SchemaFactory };
