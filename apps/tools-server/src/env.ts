import { config } from "dotenv";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const currentDir = dirname(fileURLToPath(import.meta.url));
const rootEnvPath = resolve(currentDir, "../../../.env");
const packageEnvPath = resolve(currentDir, "../.env");

if (existsSync(rootEnvPath)) {
  config({ path: rootEnvPath });
}

if (existsSync(packageEnvPath)) {
  config({ path: packageEnvPath, override: true });
}
