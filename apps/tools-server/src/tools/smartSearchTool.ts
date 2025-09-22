import { BadRequestException, Injectable, Logger } from "@nestjs/common";
import {
  ReflectiveMcpTool,
  toolField,
  type McpToolCallResult,
} from "./baseTool.js";

type SearchEngineId = "brave" | "tavily" | "exa" | "firecrawl";

type SearchToolInput = {
  query: string;
  allowSplit: boolean;
  engine: SearchEngineId;
  maxResults: number;
};

type ComponentQuery = {
  query: string;
  rationale: string;
  priority: number;
};

type SearchResultItem = {
  title: string;
  url: string;
  description: string;
  rawContent?: string;
  publishedDate?: string;
  position: number;
};

type SearchRun = {
  component: ComponentQuery;
  results: Array<SearchResultItem>;
};

type SearchToolResponse = {
  query: string;
  engine: SearchEngineId;
  allowSplit: boolean;
  executedAt: string;
  componentQueries: Array<ComponentQuery>;
  runs: Array<SearchRun>;
  summary: string;
};

class SmartWebSearchInput {
  @toolField.string({ description: "Search query to execute" })
  query!: string;

  @toolField.boolean({
    description:
      "Allow heuristic splitting of complex queries into sub-queries",
    default: true,
    required: false,
  })
  allowSplit!: boolean;

  @toolField.string({
    enum: ["brave", "tavily", "exa", "firecrawl"],
    description:
      "Preferred search engine. Falls back to first configured engine if unavailable",
    required: false,
  })
  engine?: string;

  @toolField.integer({
    description: "Maximum number of results to return per query (1-8)",
    minimum: 1,
    maximum: 8,
    required: false,
  })
  maxResults?: number;
}

interface SearchToolConfig {
  defaultEngine: SearchEngineId;
  topResults: number;
  maxComponentQueries: number;
  searchRegion: string;
  braveApiKey?: string;
  tavilyApiKey?: string;
  exaApiKey?: string;
  firecrawlApiKey?: string;
  firecrawlBaseUrl: string;
}

@Injectable()
export class SmartWebSearchTool extends ReflectiveMcpTool<
  SmartWebSearchInput,
  SearchToolInput
> {
  readonly name = "web_search" as const;
  readonly description =
    "Run a web search and summarize the top results using Brave, Tavily, Exa, or Firecrawl.";

  private readonly logger = new Logger(SmartWebSearchTool.name);
  private readonly config: SearchToolConfig;
  private readonly availableEngines: Array<SearchEngineId>;

  constructor() {
    super();
    this.config = this.resolveConfig();
    this.availableEngines = this.computeAvailableEngines();
  }

  protected readonly inputConstructor = SmartWebSearchInput;

  protected fieldOverrides() {
    return {
      maxResults: { default: this.config.topResults },
    };
  }

  protected transformInput(input: SmartWebSearchInput): SearchToolInput {
    const query = input.query.trim();
    if (!query) {
      throw new BadRequestException("Invalid payload: query is required");
    }

    const allowSplit = input.allowSplit ?? true;
    const engine = this.resolveEngine(input.engine);
    const maxResults = this.resolveMaxResults(input.maxResults);

    return { query, allowSplit, engine, maxResults };
  }

  async call(args: Record<string, unknown>): Promise<McpToolCallResult> {
    const input = this.parseArgs(args);

    const componentQueries = await this.buildComponentQueries(
      input.query,
      input.allowSplit,
    );

    const runs: Array<SearchRun> = [];
    for (const component of componentQueries) {
      try {
        const results = await this.runSearch(
          input.engine,
          component.query,
          input.maxResults,
        );
        runs.push({ component, results });
      } catch (error) {
        this.logger.error(
          `Search failed for "${component.query}" using ${input.engine}: ${String(error)}`,
        );
        runs.push({ component, results: [] });
      }
    }

    const summary = this.createRunSummary(runs);

    const response: SearchToolResponse = {
      query: input.query,
      engine: input.engine,
      allowSplit: input.allowSplit,
      executedAt: new Date().toISOString(),
      componentQueries,
      runs,
      summary,
    };

    return { content: JSON.stringify(response, null, 2) };
  }

  private resolveConfig(): SearchToolConfig {
    const env = process.env;
    const defaultEngine = this.parseEngine(
      env.SMART_SEARCH_ENGINE ?? env.WEB_SEARCH_ENGINE ?? "brave",
    );

    const topResults = this.parseIntBounded(
      env.SMART_SEARCH_TOP_N ?? env.SEARCH_TOP_N,
      3,
      1,
      8,
    );

    const maxComponentQueries = this.parseIntBounded(
      env.SMART_SEARCH_MAX_COMPONENTS ?? env.SEARCH_MAX_COMPONENTS,
      3,
      1,
      5,
    );

    return {
      defaultEngine,
      topResults,
      maxComponentQueries,
      searchRegion: env.SMART_SEARCH_REGION ?? env.SEARCH_REGION ?? "us",
      braveApiKey: env.BRAVE_SEARCH_API_KEY ?? env.BRAVE_API_KEY,
      tavilyApiKey: env.TAVILY_API_KEY,
      exaApiKey: env.EXA_API_KEY,
      firecrawlApiKey: env.FIRECRAWL_API_KEY,
      firecrawlBaseUrl:
        env.FIRECRAWL_BASE_URL?.replace(/\/$/, "") ??
        "https://api.firecrawl.dev",
    };
  }

  private computeAvailableEngines(): Array<SearchEngineId> {
    const engines: Array<SearchEngineId> = [];
    if (this.config.braveApiKey) {
      engines.push("brave");
    }
    if (this.config.tavilyApiKey) {
      engines.push("tavily");
    }
    if (this.config.exaApiKey) {
      engines.push("exa");
    }
    if (this.config.firecrawlApiKey) {
      engines.push("firecrawl");
    }
    return engines;
  }

  private resolveEngine(candidate: string | undefined): SearchEngineId {
    const preferred = this.parseEngine(candidate ?? this.config.defaultEngine);

    if (this.isEngineConfigured(preferred)) {
      return preferred;
    }

    const fallback = this.availableEngines[0];
    if (fallback) {
      this.logger.warn(
        `Engine ${preferred} not configured. Falling back to ${fallback}.`,
      );
      return fallback;
    }

    throw new BadRequestException(
      "No search engines are configured. Provide at least one API key.",
    );
  }

  private resolveMaxResults(candidate: number | undefined): number {
    return this.parseIntBounded(candidate ?? this.config.topResults, 3, 1, 8);
  }

  private parseEngine(candidate: string | undefined): SearchEngineId {
    if (!candidate) {
      return "brave";
    }
    const normalized = candidate.trim().toLowerCase();
    if (
      normalized === "tavily" ||
      normalized === "exa" ||
      normalized === "firecrawl"
    ) {
      return normalized;
    }
    return "brave";
  }

  private parseIntBounded(
    value: unknown,
    fallback: number,
    min: number,
    max: number,
  ): number {
    const parsed = Number.parseInt(String(value ?? fallback), 10);
    if (Number.isNaN(parsed)) {
      return fallback;
    }
    return Math.min(Math.max(parsed, min), max);
  }

  private isEngineConfigured(engine: SearchEngineId): boolean {
    switch (engine) {
      case "brave":
        return Boolean(this.config.braveApiKey);
      case "tavily":
        return Boolean(this.config.tavilyApiKey);
      case "exa":
        return Boolean(this.config.exaApiKey);
      case "firecrawl":
        return Boolean(this.config.firecrawlApiKey);
      default:
        return false;
    }
  }

  private async buildComponentQueries(
    query: string,
    allowSplit: boolean,
  ): Promise<Array<ComponentQuery>> {
    const components: Array<ComponentQuery> = [
      {
        query,
        rationale: "Primary query",
        priority: 1,
      },
    ];

    if (!allowSplit) {
      return components;
    }

    const additional = this.heuristicComponentQueries(query);
    for (const item of additional) {
      if (components.length >= this.config.maxComponentQueries) {
        break;
      }
      const duplicate = components.find(
        (component) =>
          component.query.toLowerCase() === item.query.toLowerCase(),
      );
      if (!duplicate) {
        components.push({
          query: item.query,
          rationale: item.rationale,
          priority: components.length + 1,
        });
      }
    }

    return components;
  }

  private heuristicComponentQueries(
    query: string,
  ): Array<Omit<ComponentQuery, "priority">> {
    const normalized = query.toLowerCase();
    const results: Array<Omit<ComponentQuery, "priority">> = [];

    const comparisonPattern = /\s+(?:vs\.?|versus|compared to|against)\s+/i;
    if (comparisonPattern.test(query)) {
      const parts = query.split(comparisonPattern).map((part) => part.trim());
      if (parts.length === 2) {
        const [a, b] = parts;
        results.push(
          { query: a, rationale: `Gather details about ${a}` },
          { query: b, rationale: `Gather details about ${b}` },
          {
            query: `${a} vs ${b} comparison`,
            rationale: `Find direct comparisons between ${a} and ${b}`,
          },
        );
      }
    }

    if (!results.length && /\sand\s|,/.test(normalized)) {
      const segments: Array<string> = query
        .split(/(?:\sand\s|,)/i)
        .map((segment) => segment.trim())
        .filter((segment): segment is string => segment.length > 0);
      if (segments.length > 1) {
        for (const topic of segments) {
          results.push({
            query: topic,
            rationale: `Explore details for "${topic}"`,
          });
        }
      }
    }

    return results;
  }

  private async runSearch(
    engine: SearchEngineId,
    query: string,
    maxResults: number,
  ): Promise<Array<SearchResultItem>> {
    switch (engine) {
      case "brave":
        return this.searchBrave(query, maxResults);
      case "tavily":
        return this.searchTavily(query, maxResults);
      case "exa":
        return this.searchExa(query, maxResults);
      case "firecrawl":
        return this.searchFirecrawl(query, maxResults);
      default:
        throw new BadRequestException("Unsupported search engine");
    }
  }

  private async searchBrave(
    query: string,
    maxResults: number,
  ): Promise<Array<SearchResultItem>> {
    if (!this.config.braveApiKey) {
      throw new BadRequestException("Brave Search API key is not configured");
    }

    const params = new URLSearchParams({
      q: query,
      count: String(maxResults),
      country: this.config.searchRegion,
    });

    const response = await fetch(
      `https://api.search.brave.com/res/v1/web/search?${params.toString()}`,
      {
        headers: {
          Accept: "application/json",
          "X-Subscription-Token": this.config.braveApiKey,
        },
      },
    );

    if (!response.ok) {
      throw new Error(`Brave search failed with status ${response.status}`);
    }

    const data = (await response.json()) as {
      web?: { results?: Array<Record<string, unknown>> };
    };

    const items = data.web?.results ?? [];
    return items.slice(0, maxResults).map((item, index) => ({
      title: this.safeString(item["title"] ?? "No title", "No title"),
      url: this.safeString(item["url"], ""),
      description: this.safeString(item["description"], ""),
      position: index + 1,
    }));
  }

  private async searchTavily(
    query: string,
    maxResults: number,
  ): Promise<Array<SearchResultItem>> {
    if (!this.config.tavilyApiKey) {
      throw new BadRequestException("Tavily API key is not configured");
    }

    const response = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.config.tavilyApiKey}`,
      },
      body: JSON.stringify({
        query,
        max_results: maxResults,
        include_raw_content: "markdown",
        search_depth: "advanced",
      }),
    });

    if (!response.ok) {
      throw new Error(`Tavily search failed with status ${response.status}`);
    }

    const data = (await response.json()) as {
      results?: Array<Record<string, unknown>>;
    };

    const items = data.results ?? [];
    return items.slice(0, maxResults).map((item, index) => ({
      title: this.safeString(item["title"], "No title"),
      url: this.safeString(item["url"], ""),
      description: this.safeString(item["content"], ""),
      rawContent:
        typeof item["raw_content"] === "string"
          ? item["raw_content"]
          : undefined,
      position: index + 1,
    }));
  }

  private async searchExa(
    query: string,
    maxResults: number,
  ): Promise<Array<SearchResultItem>> {
    if (!this.config.exaApiKey) {
      throw new BadRequestException("Exa API key is not configured");
    }

    const response = await fetch("https://api.exa.ai/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": this.config.exaApiKey,
      },
      body: JSON.stringify({
        query,
        numResults: maxResults,
        type: "keyword",
        contents: {
          text: true,
          context: true,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Exa search failed with status ${response.status}`);
    }

    const data = (await response.json()) as {
      results?: Array<Record<string, unknown>>;
    };

    const items = data.results ?? [];
    return items.slice(0, maxResults).map((item, index) => ({
      title: this.safeString(item["title"], "No title"),
      url: this.safeString(item["url"], ""),
      description: this.safeString(item["description"], ""),
      rawContent: typeof item["text"] === "string" ? item["text"] : undefined,
      publishedDate:
        typeof item["publishedDate"] === "string"
          ? item["publishedDate"]
          : undefined,
      position: index + 1,
    }));
  }

  private async searchFirecrawl(
    query: string,
    maxResults: number,
  ): Promise<Array<SearchResultItem>> {
    if (!this.config.firecrawlApiKey) {
      throw new BadRequestException("Firecrawl API key is not configured");
    }

    const response = await fetch(`${this.config.firecrawlBaseUrl}/v1/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.config.firecrawlApiKey}`,
      },
      body: JSON.stringify({
        query,
        pageOptions: { fetchPageContent: true },
      }),
    });

    if (!response.ok) {
      throw new Error(`Firecrawl search failed with status ${response.status}`);
    }

    const data = (await response.json()) as Record<string, unknown>;
    const candidateKeys: Array<"results" | "data" | "items"> = [
      "results",
      "data",
      "items",
    ];

    let rawResults: Array<Record<string, unknown>> = [];
    for (const key of candidateKeys) {
      const value = data[key];
      if (Array.isArray(value)) {
        rawResults = value as Array<Record<string, unknown>>;
        break;
      }
    }

    return rawResults.slice(0, maxResults).map((item, index) => ({
      title: this.safeString(
        item["title"] ?? item["name"] ?? item["source"],
        "No title",
      ),
      url: this.safeString(item["url"] ?? item["link"], ""),
      description: this.safeString(
        item["description"] ?? item["content"] ?? item["snippet"],
        "",
      ),
      rawContent:
        typeof item["rawContent"] === "string" ? item["rawContent"] : undefined,
      publishedDate:
        typeof item["publishedDate"] === "string"
          ? item["publishedDate"]
          : undefined,
      position: index + 1,
    }));
  }

  private createRunSummary(runs: Array<SearchRun>): string {
    if (!runs.length) {
      return "No searches were executed.";
    }

    const lines: Array<string> = [];
    for (const run of runs) {
      const resultCount = run.results.length;
      lines.push(
        `Query "${run.component.query}" → ${resultCount} result${resultCount === 1 ? "" : "s"}`,
      );
      for (const result of run.results.slice(0, 3)) {
        lines.push(`  - ${result.title} (${result.url})`);
      }
    }

    return lines.join("\n");
  }

  private safeString(value: unknown, fallback = ""): string {
    if (typeof value === "string") {
      return value;
    }
    if (typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
    return fallback;
  }
}
