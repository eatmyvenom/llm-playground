import { existsSync } from "node:fs";
import tseslintPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import prettierPlugin from "eslint-plugin-prettier";
import turboPlugin from "eslint-plugin-turbo";
import globals from "globals";

export default function createConfig({ tsconfigPath = "./tsconfig.json" } = {}) {
  const hasProjectConfig = existsSync(tsconfigPath);
  const typeAwareRules = {
    "@typescript-eslint/array-type": ["error", { default: "generic" }],
    "@typescript-eslint/consistent-type-imports": "error",
    "@typescript-eslint/explicit-function-return-type": "off",
    "@typescript-eslint/no-floating-promises": "error",
    "@typescript-eslint/no-misused-promises": "error",
    "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
    "@typescript-eslint/prefer-nullish-coalescing": "warn",
    "@typescript-eslint/restrict-template-expressions": ["warn", { allowNumber: true, allowBoolean: true }]
  };
  return [
    {
      ignores: ["dist", "node_modules", "coverage", "build"]
    },
    {
      files: ["**/*.ts", "**/*.tsx"],
      languageOptions: {
        parser: tsParser,
        parserOptions: {
          ...(hasProjectConfig
            ? { project: [tsconfigPath], tsconfigRootDir: process.cwd() }
            : {}),
          ecmaVersion: "latest",
          sourceType: "module"
        },
        globals: {
          ...globals.es2022,
          ...globals.node
        }
      },
      plugins: {
        "@typescript-eslint": tseslintPlugin,
        prettier: prettierPlugin,
        turbo: turboPlugin
      },
      rules: {
        ...(hasProjectConfig ? typeAwareRules : {}),
        "prettier/prettier": ["warn", { printWidth: 120 }],
        "turbo/no-undeclared-env-vars": "error"
      }
    }
  ];
}
