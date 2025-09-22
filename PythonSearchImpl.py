"""
title: Smart Web Search Tool
author: Vnmm
author_url: https://github.com/open-webui
description: Token-efficient web search with intelligent query analysis and splitting for comprehensive results
required_open_webui_version: 0.3.0
requirements: httpx, beautifulsoup4
version: 3.1.0
"""

import os
import json
import httpx
from typing import List, Dict, Optional, Callable, Any, Protocol
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
from abc import ABC, abstractmethod


# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------
class SearchConfig(BaseModel):
    """Configuration settings for the search tool."""
    brave_api_key: str = Field(default="", description="Your Brave Search API key")
    tavily_api_key: str = Field(default="", description="Your Tavily API key")
    exa_api_key: str = Field(default="", description="Your Exa API key")
    firecrawl_api_key: str = Field(default="", description="Your Firecrawl API key")
    firecrawl_api_base_url: str = Field(
        default="https://api.firecrawl.dev",
        description="Base URL for the Firecrawl API",
    )
    web_search_engine: str = Field(
        default="brave", 
        description="Search engine to use (brave, tavily, exa, or firecrawl)"
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for any OpenAI-compatible API"
    )
    llm_api_key: str = Field(
        default="",
        description="API key for the OpenAI-compatible provider"
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for summarization (deprecated)"
    )
    top_n_results: int = Field(
        default=3, 
        description="Number of top search results to process"
    )
    summary_model: str = Field(
        default="gpt-4.1-nano",
        description="Model to use for summarization"
    )
    max_summary_tokens: int = Field(
        default=500, 
        description="Maximum tokens for the final summary"
    )
    chunk_size: int = Field(
        default=3000, 
        description="Size of content chunks to summarize"
    )
    search_region: str = Field(default="us", description="Search region")
    debug_mode: bool = Field(default=True, description="Enable debug logging")
    enable_query_analysis: bool = Field(
        default=True, 
        description="Enable LLM-powered query analysis and splitting"
    )
    query_analysis_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for query analysis"
    )
    max_component_queries: int = Field(
        default=3,
        description="Maximum number of component queries to split into"
    )


# --------------------------------------------------------------------------------------
# DATA MODELS
# --------------------------------------------------------------------------------------
class ComponentQuery(BaseModel):
    """Represents a component search query."""
    query: str = Field(description="The specific search query")
    rationale: str = Field(description="Why this component is needed")
    priority: int = Field(description="Priority order (1 = highest priority)", ge=1, le=10)


class QueryAnalysis(BaseModel):
    """Structured output for query analysis."""
    should_split: bool = Field(description="Whether the query should be split")
    reasoning: str = Field(description="Explanation of the decision")
    component_queries: List[ComponentQuery] = Field(
        default_factory=list,
        description="List of component queries if splitting is needed"
    )
    original_query_sufficient: bool = Field(
        default=True,
        description="Whether the original query alone would be sufficient"
    )


class SearchResult(BaseModel):
    """Represents a single search result."""
    title: str
    url: str
    description: str
    raw_content: Optional[str] = None
    published_date: Optional[str] = None
    position: Optional[int] = None


class SummaryResponse(BaseModel):
    """Structured output for content summarization."""
    summary: str = Field(description="The summarized content")
    quality: int = Field(description="Quality rating from 1-5, with 5 being highest quality and most relevant to the query", ge=1, le=5)


class ProcessedSource(BaseModel):
    """Represents a processed source with summary."""
    title: str
    url: str
    summary: str
    position: int
    published_date: Optional[str] = None
    quality: int = Field(default=3, description="Quality rating from 1-5")


# --------------------------------------------------------------------------------------
# PROTOCOLS AND INTERFACES
# --------------------------------------------------------------------------------------
class EventEmitter(Protocol):
    """Protocol for event emission."""
    async def __call__(self, event: Dict[str, Any]) -> None: ...


class StatusEmitter:
    """Handles status message emission."""
    
    def __init__(self, event_emitter: Optional[EventEmitter] = None):
        self.event_emitter = event_emitter
    
    async def emit(self, message: str, done: bool = False) -> None:
        """Emit a status message."""
        if self.event_emitter:
            await self.event_emitter({
                "type": "status",
                "data": {"description": message, "done": done}
            })


class SearchEngine(ABC):
    """Abstract base class for search engines."""
    
    def __init__(self, config: SearchConfig, status_emitter: StatusEmitter):
        self.config = config
        self.status_emitter = status_emitter
    
    @abstractmethod
    async def search(self, query: str) -> List[SearchResult]:
        """Perform a search and return results."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the search engine is properly configured."""
        pass


# --------------------------------------------------------------------------------------
# SEARCH ENGINE IMPLEMENTATIONS
# --------------------------------------------------------------------------------------
class BraveSearchEngine(SearchEngine):
    """Brave Search implementation."""
    
    def is_configured(self) -> bool:
        return bool(self.config.brave_api_key and self.config.brave_api_key.strip())
    
    async def search(self, query: str) -> List[SearchResult]:
        if not self.is_configured():
            await self.status_emitter.emit(
                "❌ Error: Brave Search API key not configured"
            )
            return []
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.config.brave_api_key,
        }
        params = {
            "q": query,
            "count": self.config.top_n_results,
            "country": self.config.search_region,
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers=headers,
                    params=params,
                    timeout=10,
                )
                response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "web" in data and "results" in data["web"]:
                for i, result in enumerate(data["web"]["results"][:self.config.top_n_results], 1):
                    results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("url", ""),
                        description=result.get("description", "No description"),
                        position=i
                    ))
            
            return results
            
        except Exception as e:
            await self.status_emitter.emit(f"❌ Brave search error: {str(e)}")
            return []


class TavilySearchEngine(SearchEngine):
    """Tavily Search implementation."""
    
    def is_configured(self) -> bool:
        return bool(self.config.tavily_api_key and self.config.tavily_api_key.strip())
    
    async def search(self, query: str) -> List[SearchResult]:
        if not self.is_configured():
            await self.status_emitter.emit(
                "❌ Error: Tavily API key not configured"
            )
            return []
        
        headers = {
            "Authorization": f"Bearer {self.config.tavily_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "max_results": self.config.top_n_results,
            "include_raw_content": "markdown",
            "search_depth": "advanced"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "results" in data:
                for i, result in enumerate(data["results"], 1):
                    results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("url", ""),
                        description=result.get("content", "No description"),
                        raw_content=result.get("raw_content", ""),
                        position=i
                    ))
            
            return results
            
        except Exception as e:
            await self.status_emitter.emit(f"❌ Tavily search error: {str(e)}")
            return []


class ExaSearchEngine(SearchEngine):
    """Exa Search implementation."""
    
    def is_configured(self) -> bool:
        return bool(self.config.exa_api_key and self.config.exa_api_key.strip())
    
    async def search(self, query: str) -> List[SearchResult]:
        if not self.is_configured():
            await self.status_emitter.emit(
                "❌ Error: Exa API key not configured"
            )
            return []
        
        headers = {
            "x-api-key": self.config.exa_api_key,
            "Content-Type": "application/json",
        }
        
        payload = {
            "query": query,
            "numResults": self.config.top_n_results,
            "type": "keyword",
            "contents": {
                "text": True,
                "context": True,
            }
        }
        
        if self.config.summary_model == "auto":
            payload["contents"] = {
                "summary": {
                    "query": "Summarize content with very high detail. Include all relevant information."
                }
            }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.exa.ai/search",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "results" in data:
                for i, result in enumerate(data["results"], 1):
                    results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("url", ""),
                        description=result.get("description", "No description"),
                        published_date=result.get("publishedDate"),
                        raw_content=result.get("text"),
                        position=i
                    ))
            
            return results
            
        except Exception as e:
            await self.status_emitter.emit(f"❌ Exa search error: {str(e)}")
            return []


class FirecrawlSearchEngine(SearchEngine):
    """Firecrawl Search implementation."""

    def is_configured(self) -> bool:
        return bool(self.config.firecrawl_api_key and self.config.firecrawl_api_key.strip())

    async def search(self, query: str) -> List[SearchResult]:
        if not self.is_configured():
            await self.status_emitter.emit(
                "❌ Error: Firecrawl API key not configured"
            )
            return []

        base_url = (self.config.firecrawl_api_base_url or "https://api.firecrawl.dev").rstrip("/")
        search_url = f"{base_url}/v1/search"

        headers = {
            "Authorization": f"Bearer {self.config.firecrawl_api_key}",
            "Content-Type": "application/json",
        }

        candidate_payloads = [
            {"query": query, "pageOptions": {"fetchPageContent": True}},
            {"query": query},
        ]

        try:
            response_data = None
            async with httpx.AsyncClient() as client:
                last_error: Optional[Exception] = None
                for payload in candidate_payloads:
                    try:
                        response = await client.post(
                            search_url,
                            headers=headers,
                            json=payload,
                            timeout=30,
                        )
                        response.raise_for_status()
                        response_data = response.json()
                        break
                    except httpx.HTTPStatusError as http_err:
                        last_error = http_err
                        status_code = getattr(http_err.response, "status_code", None)
                        if status_code in {400, 404, 422}:
                            continue
                        raise
                    except Exception as generic_error:
                        last_error = generic_error
                        raise

            if response_data is None:
                if last_error:
                    raise last_error
                return []

            data = response_data
            raw_results = (
                data.get("results")
                or data.get("data")
                or data.get("items")
                or []
            )

            results: List[SearchResult] = []

            for i, result in enumerate(raw_results[: self.config.top_n_results], 1):
                title = result.get("title") or result.get("name") or "No title"
                url = (
                    result.get("url")
                    or result.get("link")
                    or result.get("source")
                    or ""
                )
                description = (
                    result.get("description")
                    or result.get("content")
                    or result.get("snippet")
                    or ""
                )
                raw_content = (
                    result.get("rawContent")
                    or result.get("raw_content")
                    or result.get("content")
                    or ""
                )
                published_date = (
                    result.get("publishedDate")
                    or result.get("published_date")
                    or result.get("date")
                )

                results.append(SearchResult(
                    title=title,
                    url=url,
                    description=description,
                    raw_content=raw_content,
                    published_date=published_date,
                    position=i,
                ))

            return results

        except Exception as e:
            await self.status_emitter.emit(f"❌ Firecrawl search error: {str(e)}")
            return []


# --------------------------------------------------------------------------------------
# SEARCH ENGINE FACTORY
# --------------------------------------------------------------------------------------
class SearchEngineFactory:
    """Factory for creating search engines."""
    
    @staticmethod
    def create_engine(engine_type: str, config: SearchConfig, status_emitter: StatusEmitter) -> SearchEngine:
        """Create a search engine instance based on type."""
        engines = {
            "brave": BraveSearchEngine,
            "tavily": TavilySearchEngine,
            "exa": ExaSearchEngine,
            "firecrawl": FirecrawlSearchEngine,
        }
        
        engine_class = engines.get(engine_type.lower())
        if not engine_class:
            raise ValueError(f"Unsupported search engine: {engine_type}")
        
        return engine_class(config, status_emitter)


# --------------------------------------------------------------------------------------
# QUERY ANALYZER
# --------------------------------------------------------------------------------------
class QueryAnalyzer:
    """Analyzes search queries and determines if they should be split."""
    
    def __init__(self, config: SearchConfig, status_emitter: StatusEmitter):
        self.config = config
        self.status_emitter = status_emitter
    
    def _get_api_key(self) -> Optional[str]:
        """Get the API key for LLM calls."""
        return (
            self.config.llm_api_key
            or self.config.openai_api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    
    def _get_llm_base_url(self) -> str:
        """Get the base URL for LLM API."""
        return (self.config.llm_base_url or "https://api.openai.com/v1").rstrip("/")
    
    async def analyze_query(
        self,
        query: str,
        user_messages: Optional[List[Dict[str, str]]] = None,
    ) -> QueryAnalysis:
        """Analyze a search query to determine if it should be split."""
        if not self.config.enable_query_analysis:
            return QueryAnalysis(
                should_split=False,
                reasoning="Query analysis is disabled in configuration",
                original_query_sufficient=True
            )
        
        api_key = self._get_api_key()
        if not api_key:
            await self.status_emitter.emit("⚠️ No LLM API key - skipping query analysis")
            return QueryAnalysis(
                should_split=False,
                reasoning="No LLM API key available for query analysis",
                original_query_sufficient=True
            )
        
        await self.status_emitter.emit("🧠 Analyzing search query...")
        
        try:
            user_context = ""
            if user_messages:
                recent_messages = user_messages[-3:]  # Last 3 messages for context
                user_context = "\n".join([
                    f"- {msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in recent_messages
                ])
            
            system_prompt = """You are a search query analyzer. Your job is to determine if a complex search query should be split into multiple component searches to get better, more comprehensive results.

Guidelines for splitting:
1. Split if the query contains multiple distinct topics or concepts that would benefit from separate searches
2. Split if the query asks for comparisons between different entities/products/concepts
3. Split if the query contains "vs", "versus", "compared to", "differences between"
4. Split if the query asks for multiple types of information that are unrelated
5. DO NOT split if the query is about a single concept, even if complex
6. DO NOT split simple queries or questions about one specific thing
7. DO NOT split in order to help by gathering more information than what the search query requests.
8. Assume that multiple searches have taken place already if the query is highly targeted or specific.

Return a structured JSON response following the QueryAnalysis schema."""
            
            user_prompt = f"""
Analyze this search query: "{query}"

User context (recent conversation):
{user_context}

Should this query be split into multiple component searches? If yes, provide up to {self.config.max_component_queries} component queries with clear rationale and priority.

Provide your analysis in valid JSON format matching the QueryAnalysis schema.
"""
            
            # Create the response format schema
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "query_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "should_split": {"type": "boolean"},
                            "reasoning": {"type": "string"},
                            "component_queries": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "rationale": {"type": "string"},
                                        "priority": {"type": "integer", "minimum": 1, "maximum": 10}
                                    },
                                    "required": ["query", "rationale", "priority"],
                                    "additionalProperties": False
                                }
                            },
                            "original_query_sufficient": {"type": "boolean"}
                        },
                        "required": ["should_split", "reasoning", "component_queries", "original_query_sufficient"],
                        "additionalProperties": False
                    }
                }
            }
            
            chat_url = f"{self._get_llm_base_url()}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            async with httpx.AsyncClient() as client:
                # Try with structured output first
                try:
                    response = await client.post(
                        chat_url,
                        headers=headers,
                        json={
                            "model": self.config.query_analysis_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.1,
                            "max_tokens": 1000,
                            "response_format": response_format,
                        },
                        timeout=15,
                    )
                    response.raise_for_status()
                except Exception as structured_error:
                    # Fallback to regular chat completion if structured output fails
                    await self.status_emitter.emit("⚠️ Structured output not supported, using fallback")
                    response = await client.post(
                        chat_url,
                        headers=headers,
                        json={
                            "model": self.config.query_analysis_model,
                            "messages": [
                                {"role": "system", "content": system_prompt + "\n\nIMPORTANT: Respond with valid JSON only, no additional text."},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.1,
                            "max_tokens": 1000,
                        },
                        timeout=15,
                    )
                    response.raise_for_status()
            
            data = response.json()
            analysis_json = json.loads(data["choices"][0]["message"]["content"])
            
            # Create QueryAnalysis object
            component_queries = [
                ComponentQuery(**comp) for comp in analysis_json.get("component_queries", [])
            ]
            
            analysis = QueryAnalysis(
                should_split=analysis_json["should_split"],
                reasoning=analysis_json["reasoning"],
                component_queries=component_queries,
                original_query_sufficient=analysis_json.get("original_query_sufficient", True)
            )
            
            if analysis.should_split:
                await self.status_emitter.emit(
                    f"🔀 Query will be split into {len(analysis.component_queries)} components"
                )
            else:
                await self.status_emitter.emit("✅ Query analysis complete - no splitting needed")
            
            return analysis
            
        except Exception as e:
            await self.status_emitter.emit(f"⚠️ Query analysis failed: {str(e)} - using original query")
            return QueryAnalysis(
                should_split=False,
                reasoning=f"Query analysis failed: {str(e)}",
                original_query_sufficient=True
            )


# --------------------------------------------------------------------------------------
# CONTENT EXTRACTION
# --------------------------------------------------------------------------------------
class ContentExtractor:
    """Handles content extraction from URLs."""
    
    def __init__(self, config: SearchConfig, status_emitter: StatusEmitter):
        self.config = config
        self.status_emitter = status_emitter
    
    async def extract_content(self, url: str) -> str:
        """Extract content from a URL using the configured provider."""
        engine = (self.config.web_search_engine or "").lower()

        if engine == "firecrawl":
            firecrawl_content = await self._extract_with_firecrawl(url)
            if firecrawl_content:
                return firecrawl_content
            await self.status_emitter.emit(
                "⚠️ Firecrawl extraction failed - attempting Tavily fallback"
            )

        if not self.config.tavily_api_key:
            await self.status_emitter.emit("⚠️ No Tavily API key - skipping content extraction")
            return ""
        
        payload = {
            "urls": [url],
            "include_images": False,
            "include_raw_content": True,
        }
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.tavily_api_key}",
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/extract",
                    headers=headers,
                    json=payload,
                    timeout=15,
                )
                response.raise_for_status()
            
            data = response.json()
            content = ""
            
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                content = (
                    result.get("raw_content", "")
                    or result.get("content", "")
                    or result.get("text", "")
                )
            
            if content:
                content = content[:self.config.chunk_size]
                await self.status_emitter.emit(
                    f"✅ Extracted {len(content)} characters"
                )
            else:
                await self.status_emitter.emit("❌ No content found in Tavily response")
            
            return content
            
        except Exception as e:
            await self.status_emitter.emit(f"❌ Content extraction error: {str(e)}")
            return ""

    async def _extract_with_firecrawl(self, url: str) -> str:
        """Extract content using the Firecrawl API."""
        if not self.config.firecrawl_api_key:
            await self.status_emitter.emit("⚠️ No Firecrawl API key - skipping content extraction")
            return ""

        base_url = (self.config.firecrawl_api_base_url or "https://api.firecrawl.dev").rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.firecrawl_api_key}",
        }

        endpoints = [
            (f"{base_url}/v1/extract", {"url": url, "pageOptions": {"fetchPageContent": True}}),
            (f"{base_url}/v1/crawl", {"url": url, "maxDepth": 0, "formats": ["markdown"]}),
        ]

        async with httpx.AsyncClient() as client:
            for endpoint, payload in endpoints:
                try:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )

                    if response.status_code == 404:
                        continue

                    response.raise_for_status()
                    data = response.json()
                    content = self._parse_firecrawl_content(data)

                    if content:
                        content = content[: self.config.chunk_size]
                        await self.status_emitter.emit(
                            f"✅ Extracted {len(content)} characters via Firecrawl"
                        )
                        return content

                except Exception as e:
                    await self.status_emitter.emit(
                        f"⚠️ Firecrawl extraction error ({endpoint}): {str(e)}"
                    )

        await self.status_emitter.emit("❌ Firecrawl extraction failed")
        return ""

    @staticmethod
    def _parse_firecrawl_content(data: Any) -> str:
        """Extract text content from a Firecrawl response payload."""
        candidates: List[str] = []

        def collect_from_dict(obj: Dict[str, Any]) -> None:
            for key in ("markdown", "content", "text", "rawContent", "raw_content"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value)

        if isinstance(data, dict):
            collect_from_dict(data)
            inner = data.get("data")
            if isinstance(inner, dict):
                collect_from_dict(inner)
            elif isinstance(inner, list) and inner:
                first = inner[0]
                if isinstance(first, dict):
                    collect_from_dict(first)

            results = data.get("results") or data.get("items")
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    collect_from_dict(first)

        elif isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                collect_from_dict(first)

        for candidate in candidates:
            if candidate.strip():
                return candidate

        return ""


# --------------------------------------------------------------------------------------
# SUMMARIZATION
# --------------------------------------------------------------------------------------
class ContentSummarizer:
    """Handles content summarization using LLM."""
    
    def __init__(self, config: SearchConfig, status_emitter: StatusEmitter):
        self.config = config
        self.status_emitter = status_emitter
    
    def _get_api_key(self) -> Optional[str]:
        """Get the API key for LLM calls."""
        return (
            self.config.llm_api_key
            or self.config.openai_api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    
    def _get_llm_base_url(self) -> str:
        """Get the base URL for LLM API."""
        return (self.config.llm_base_url or "https://api.openai.com/v1").rstrip("/")
    
    def _create_fallback_summary(self, content: str, query: str) -> Dict[str, Any]:
        """Create a simple fallback summary when LLM is unavailable."""
        sentences = content.split(". ")
        relevant_sentences = []
        query_words = query.lower().split()
        
        for sentence in sentences[:10]:
            if any(word in sentence.lower() for word in query_words):
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= 3:
                    break
        
        if relevant_sentences:
            summary = ". ".join(relevant_sentences)
            summary = summary[:300] + "..." if len(summary) > 300 else summary
        else:
            summary = content[:200] + "..."
        
        # Default quality for fallback summaries
        return {
            "summary": summary,
            "quality": 3
        }
    
    async def summarize_content(
        self,
        content: str,
        source_title: str,
        source_url: str,
        query: str,
        user_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Summarize content using LLM or fallback method."""
        api_key = self._get_api_key()
        
        if not api_key:
            await self.status_emitter.emit("📝 Creating simple summary (no LLM key)")
            return self._create_fallback_summary(content, query)
        
        try:
            user_prompt_text = user_messages[-1]["content"] if user_messages else ""
            
            system_prompt = """You are a content summarizer. Your job is to summarize web content and rate its quality relative to the search query.

Guidelines for quality rating:
- 5: Excellent - Directly answers the query with comprehensive, accurate information
- 4: Good - Relevant to the query with useful information, minor gaps
- 3: Fair - Somewhat relevant with some useful information
- 2: Poor - Minimally relevant, limited useful information
- 1: Very Poor - Not relevant or contains mostly irrelevant information

Return a structured JSON response following the SummaryResponse schema."""
            
            user_prompt = f"""
Summarize this web content and rate its quality relative to the search query.

Search Query: "{query}"
User Context: "{user_prompt_text}"
Source Title: "{source_title}"
Source URL: "{source_url}"

Content to summarize:
{content}

Provide:
1. A comprehensive summary in markdown format
2. A quality rating (1-5) based on relevance and usefulness to the query

Respond with valid JSON matching the SummaryResponse schema.
"""
            
            # Create the response format schema
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "summary_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "quality": {"type": "integer", "minimum": 1, "maximum": 5}
                        },
                        "required": ["summary", "quality"],
                        "additionalProperties": False
                    }
                }
            }
            
            chat_url = f"{self._get_llm_base_url()}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            async with httpx.AsyncClient() as client:
                # Try with structured output first
                try:
                    response = await client.post(
                        chat_url,
                        headers=headers,
                        json={
                            "model": self.config.summary_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.3,
                            "max_tokens": self.config.max_summary_tokens,
                            "response_format": response_format,
                        },
                        timeout=20,
                    )
                    response.raise_for_status()
                except Exception as structured_error:
                    # Fallback to regular chat completion if structured output fails
                    await self.status_emitter.emit("⚠️ Structured output not supported, using fallback")
                    response = await client.post(
                        chat_url,
                        headers=headers,
                        json={
                            "model": self.config.summary_model,
                            "messages": [
                                {"role": "system", "content": system_prompt + "\n\nIMPORTANT: Respond with valid JSON only, no additional text."},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.3,
                            "max_tokens": self.config.max_summary_tokens,
                        },
                        timeout=20,
                    )
                    response.raise_for_status()
            
            data = response.json()
            summary_json = json.loads(data["choices"][0]["message"]["content"])
            
            result = {
                "summary": summary_json["summary"],
                "quality": summary_json["quality"]
            }
            
            await self.status_emitter.emit(f"✅ Created AI summary (quality: {result['quality']}/5)")
            return result
            
        except Exception as e:
            await self.status_emitter.emit(f"⚠️ LLM error, using fallback: {str(e)}")
            return self._create_fallback_summary(content, query)


# --------------------------------------------------------------------------------------
# CITATION MANAGER
# --------------------------------------------------------------------------------------
class CitationManager:
    """Handles citation creation and emission."""

    # Citations for each message stored by message ID and list of citations {message_id1: [citation1, citation2], message_id2: [citation3, ...]}
    citations: Dict[str, List[Dict[str, str]]] = {}

    def __init__(self, event_emitter: Optional[EventEmitter] = None):
        self.event_emitter = event_emitter
    
    async def emit_citation(
        self,
        title: str,
        url: str,
        quality: int,
        description: str,
        summary: str = "",
        message_id: str = None,
    ) -> None:
        """Emit a citation event."""
        if not self.event_emitter:
            return
        
        content = f"Quality: {quality}/5\n\n{description}"
        if summary:
            content += f"\n\nSummary: {summary}"

        citation = {
            "type": "citation",
            "data": {
                "document": [content],
                "metadata": [{
                    "date_accessed": datetime.now().isoformat(),
                    "source": url,
                }],
                "source": {
                    "name": title,
                    "url": url,
                },
            },
        }

        if message_id and message_id.strip():  # Check for both None and empty/whitespace strings
            if message_id not in self.citations:
                self.citations[message_id] = []
            self.citations[message_id].append(citation["data"])
            print(f"Citation stored for message_id: {message_id}")
        else:
            print(f"No valid message ID provided for citation. Received: {repr(message_id)}")
            # You might want to generate a fallback message_id or handle this case differently
            # For now, we'll just emit the citation without storing it

        await self.event_emitter(citation)

    def gather_citations(self, message_id: str) -> List[Dict[str, str]]:
        """Gather citations for a specific message ID."""
        return self.citations.get(message_id, [])


# --------------------------------------------------------------------------------------
# RESULT FORMATTER
# --------------------------------------------------------------------------------------
class ResultFormatter:
    """Handles formatting of search results."""
    
    @staticmethod
    def create_final_summary(query: str, sources: List[ProcessedSource]) -> str:
        """Create the final summary JSON."""
        source_data = [source.dict() for source in sources]
        return json.dumps({"query": query, "sources": source_data}, indent=2)
    
    @staticmethod
    def format_search_results(results: List[SearchResult]) -> str:
        """Format search results for display."""
        formatted = []
        for result in results[:5]:
            formatted.append(
                f"{result.position or 'N/A'}. **{result.title}**\n"
                f"   {result.url}\n"
                f"   {result.description[:100]}..."
            )
        return "\n\n".join(formatted)


# --------------------------------------------------------------------------------------
# API TESTER
# --------------------------------------------------------------------------------------
class ApiTester:
    """Tests API connectivity and configuration."""
    
    def __init__(self, config: SearchConfig):
        self.config = config
    
    def _has_key(self, key: Optional[str]) -> bool:
        """Check if a key is valid."""
        return bool(key and isinstance(key, str) and key.strip())
    
    async def test_all_apis(self) -> str:
        """Test all configured APIs."""
        results = []
        
        async with httpx.AsyncClient() as client:
            # Test Brave Search
            results.append(await self._test_brave(client))
            # Test Tavily
            results.append(await self._test_tavily(client))
            # Test Firecrawl
            results.append(await self._test_firecrawl(client))
            # Test OpenAI
            results.append(await self._test_openai(client))

        return "## 🔧 API Test Results\n\n" + "\n".join(results)
    
    async def _test_brave(self, client: httpx.AsyncClient) -> str:
        """Test Brave Search API."""
        if not self._has_key(self.config.brave_api_key):
            return "❌ Brave Search API: No API key configured"
        
        try:
            headers = {"X-Subscription-Token": self.config.brave_api_key}
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search?q=test&count=1",
                headers=headers,
                timeout=5,
            )
            return "✅ Brave Search API: Working" if response.status_code == 200 else f"❌ Brave Search API: Status {response.status_code}"
        except Exception as e:
            return f"❌ Brave Search API: {str(e)}"
    
    async def _test_tavily(self, client: httpx.AsyncClient) -> str:
        """Test Tavily API."""
        if not self._has_key(self.config.tavily_api_key):
            return "❌ Tavily API: No API key configured"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.tavily_api_key}",
            }
            response = await client.post(
                "https://api.tavily.com/extract",
                json={"urls": ["https://example.com"]},
                headers=headers,
                timeout=5,
            )
            return "✅ Tavily API: Working" if response.status_code in [200, 422] else f"❌ Tavily API: Status {response.status_code}"
        except Exception as e:
            return f"❌ Tavily API: {str(e)}"

    async def _test_firecrawl(self, client: httpx.AsyncClient) -> str:
        """Test Firecrawl API."""
        if not self._has_key(self.config.firecrawl_api_key):
            return "❌ Firecrawl API: No API key configured"

        base_url = (self.config.firecrawl_api_base_url or "https://api.firecrawl.dev").rstrip("/")

        try:
            response = await client.post(
                f"{base_url}/v1/search",
                headers={
                    "Authorization": f"Bearer {self.config.firecrawl_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": "Open WebUI health check",
                    "pageOptions": {"fetchPageContent": False},
                },
                timeout=5,
            )
            return "✅ Firecrawl API: Working" if response.status_code == 200 else f"❌ Firecrawl API: Status {response.status_code}"
        except Exception as e:
            return f"❌ Firecrawl API: {str(e)}"

    async def _test_openai(self, client: httpx.AsyncClient) -> str:
        """Test OpenAI API."""
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self._has_key(api_key):
            return "⚠️ OpenAI API: No API key (will use fallback summaries)"
        
        try:
            response = await client.post(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5,
            )
            return "✅ OpenAI API: Working" if response.status_code == 200 else f"❌ OpenAI API: Status {response.status_code}"
        except Exception as e:
            return f"❌ OpenAI API: {str(e)}"


# --------------------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# --------------------------------------------------------------------------------------
class SmartSearchOrchestrator:
    """Main orchestrator that coordinates all search operations."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.status_emitter = None
        self.citation_manager = None
        self._cache: Dict[str, Any] = {}
    
    def _setup_components(self, event_emitter: Optional[EventEmitter]) -> None:
        """Setup all components with the event emitter."""
        self.status_emitter = StatusEmitter(event_emitter)
        self.citation_manager = CitationManager(event_emitter)

    async def read_web_page(
        self,
        url: str,
        query: str,
        summarize: bool = True,
        event_emitter: Optional[EventEmitter] = None,
        user_messages: Optional[List[Dict[str, str]]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """Read a web page and return its content."""

        self._setup_components(event_emitter)

        extractor_url = "https://api.tavily.com/extract"
        extractor_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.tavily_api_key}",
        }
        extractor_payload = {
            "urls": [url],
            "extract_depth": "advanced"
        }

        if self.config.web_search_engine == "exa":
            extractor_url = "https://api.exa.ai/contents"
            del extractor_headers["Authorization"]  # Remove Authorization header for Exa
            extractor_headers["x-api-key"] = self.config.exa_api_key
            extractor_payload = {
                "ids": [url],
                "text": True
            }
        elif self.config.web_search_engine == "firecrawl":
            if not self.config.firecrawl_api_key:
                await self.status_emitter.emit("⚠️ Firecrawl API key not configured for content extraction")
                return ""

            base_url = (self.config.firecrawl_api_base_url or "https://api.firecrawl.dev").rstrip("/")
            extractor_url = f"{base_url}/v1/extract"
            extractor_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.firecrawl_api_key}",
            }
            extractor_payload = {
                "url": url,
                "pageOptions": {"fetchPageContent": True},
                "formats": ["markdown", "rawContent"],
            }

        await self.status_emitter.emit(f"🌐 Extracting content from: {url}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    extractor_url,
                    headers=extractor_headers,
                    json=extractor_payload,
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()

                    content = ""
                    if self.config.web_search_engine == "exa":
                        content = data.get("results", [{}])[0].get("text")
                    elif self.config.web_search_engine == "firecrawl":
                        content = ContentExtractor._parse_firecrawl_content(data)
                    else:
                        content = data.get("results", [{}])[0].get("raw_content")

                    if not content or len(content) < 50:
                        await self.status_emitter.emit("❌ Extracted content is too short or empty")
                        return ""

                    await self.status_emitter.emit(f"✅ Extracted {len(content)} characters from web page")

                    if not summarize:
                        await self.status_emitter.emit("📝 Returning raw content (no summarization requested)")
                        # Create citation for raw content
                        if self.citation_manager:
                            await self.citation_manager.emit_citation(
                                "Web Page Content", url, 5, "Full content extracted from web page.", content, message_id
                            )
                        return {"summary": content, "quality": 5}

                    summarizer = ContentSummarizer(self.config, self.status_emitter)
                    summary = await summarizer.summarize_content(
                        content, "", url, query, user_messages
                    )

                    return summary

                else:
                    await self.status_emitter.emit(f"❌ Failed to extract content: {response.status_code}")
                    return ""

            except Exception as e:
                await self.status_emitter.emit(f"❌ Error occurred while extracting content: {str(e)}")
                # Fallback: try to get content directly
                try:
                    response = await client.get(url, timeout=30)
                    response.raise_for_status()
                    return response.text[:1000] + "..." if len(response.text) > 1000 else response.text
                except Exception as fallback_e:
                    await self.status_emitter.emit(f"❌ Fallback failed: {str(fallback_e)}")
                    return ""

    async def search_and_summarize(
        self,
        query: str,
        allow_split: bool = True,
        event_emitter: Optional[EventEmitter] = None,
        user_messages: Optional[List[Dict[str, str]]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """Main search and summarization workflow with intelligent query analysis."""
        self._setup_components(event_emitter)

        if not allow_split:
            # If splitting is not allowed, we can proceed with a single query
            return await self._handle_single_search(query, user_messages, message_id)

        try:
            # Step 1: Analyze query to determine if splitting is needed
            query_analyzer = QueryAnalyzer(self.config, self.status_emitter)
            analysis = await query_analyzer.analyze_query(query, user_messages)
            
            if analysis.should_split and analysis.component_queries:
                return await self._handle_split_search(query, analysis, user_messages, message_id)
            else:
                return await self._handle_single_search(query, user_messages, message_id)
            
        except Exception as e:
            error_msg = f"❌ **Error in search process:** {str(e)}"
            await self.status_emitter.emit(error_msg, done=True)
            return error_msg
    
    async def _handle_single_search(
        self,
        query: str,
        user_messages: Optional[List[Dict[str, str]]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """Handle a single search query."""
        await self.status_emitter.emit(f"🔍 Performing single search for: {query}")
        
        search_engine = SearchEngineFactory.create_engine(
            self.config.web_search_engine, self.config, self.status_emitter
        )
        search_results = await search_engine.search(query)
        
        await self.status_emitter.emit(f"📊 Found {len(search_results)} search results")
        
        if not search_results:
            return "❌ No search results found from Web Search."

        processed_sources = await self._process_sources(search_results, query, user_messages, message_id)

        for i, source in enumerate(processed_sources, 1):
            source.position = i
        
        if not processed_sources:
            return self._create_error_response(search_results)
        
        final_summary = ResultFormatter.create_final_summary(query, processed_sources)
        await self.status_emitter.emit("✅ Search completed successfully!", done=True)

        return final_summary

    async def _handle_split_search(
        self,
        original_query: str,
        analysis: QueryAnalysis,
        user_messages: Optional[List[Dict[str, str]]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """Handle multiple component searches and combine results."""
        await self.status_emitter.emit(
            f"🔀 Executing {len(analysis.component_queries)} component searches"
        )
        
        search_engine = SearchEngineFactory.create_engine(
            self.config.web_search_engine, self.config, self.status_emitter
        )
        
        all_processed_sources = []
        
        # Sort component queries by priority
        sorted_queries = sorted(analysis.component_queries, key=lambda x: x.priority)
        
        for i, component in enumerate(sorted_queries, 1):
            await self.status_emitter.emit(
                f"� Component search {i}/{len(sorted_queries)}: {component.query}"
            )
            
            try:
                # Perform search for this component
                search_results = await search_engine.search(component.query)
                
                if search_results:
                    # Process sources for this component
                    processed_sources = await self._process_sources(
                        search_results, component.query, user_messages, message_id
                    )
                    
                    # Add component context to sources
                    for source in processed_sources:
                        source.summary = f"**[Component: {component.query}]**\n{source.summary}"
                    
                    all_processed_sources.extend(processed_sources)
                    await self.status_emitter.emit(
                        f"✅ Component {i} completed: {len(processed_sources)} sources processed"
                    )
                else:
                    await self.status_emitter.emit(f"⚠️ Component {i}: No results found")
                    
            except Exception as e:
                await self.status_emitter.emit(f"❌ Error in component {i}: {str(e)}")
                continue
        
        if not all_processed_sources:
            return f"❌ No results found for any component searches.\n\nOriginal query: {original_query}\nComponent queries attempted: {[c.query for c in sorted_queries]}"
        
        # Renumber positions sequentially across all component searches
        for i, source in enumerate(all_processed_sources, 1):
            source.position = i
        
        # Create comprehensive summary
        final_summary = self._create_combined_summary(
            original_query, analysis, all_processed_sources
        )
        
        await self.status_emitter.emit(
            f"✅ Multi-component search completed! Total sources: {len(all_processed_sources)}", 
            done=True
        )
        
        # Emit final citation
        if self.citation_manager:
            await self.citation_manager.emit_citation(
                "SmartSearch Tool (Multi-Component)", "", 0, "Multi-component search results compiled", final_summary, message_id
            )
        
        return final_summary
    
    def _create_combined_summary(
        self,
        original_query: str,
        analysis: QueryAnalysis,
        all_sources: List[ProcessedSource],
    ) -> str:
        """Create a combined summary for multi-component search."""
        source_data = [source.dict() for source in all_sources]
        
        combined_data = {
            "original_query": original_query,
            "analysis": {
                "reasoning": analysis.reasoning,
                "component_queries": [
                    {"query": c.query, "rationale": c.rationale, "priority": c.priority}
                    for c in analysis.component_queries
                ]
            },
            "total_sources": len(all_sources),
            "sources": source_data
        }
        
        return json.dumps(combined_data, indent=2)
    
    async def _process_sources(
        self,
        search_results: List[SearchResult],
        query: str,
        user_messages: Optional[List[Dict[str, str]]],
        message_id: Optional[str] = None,
    ) -> List[ProcessedSource]:
        """Process search results into summarized sources."""
        content_extractor = ContentExtractor(self.config, self.status_emitter)
        content_summarizer = ContentSummarizer(self.config, self.status_emitter)
        
        async def process_single_source(result: SearchResult) -> Optional[ProcessedSource]:
            await self.status_emitter.emit(
                f"📄 Processing source {result.position}/{len(search_results)}: {result.title[:50]}..."
            )

            # Handle pre-existing summary (Exa)
            if hasattr(result, 'summary') and getattr(result, 'summary'):
                summary = getattr(result, 'summary')
                await self.citation_manager.emit_citation(
                    result.title, result.url, 3, result.description, summary, message_id
                )
                return ProcessedSource(
                    title=result.title,
                    url=result.url,
                    summary=summary,
                    position=result.position or 0,
                    published_date=result.published_date,
                    quality=3  # Default quality for pre-existing summaries
                )
            
            # Extract content
            content = result.raw_content or await content_extractor.extract_content(result.url)
            
            if not content or len(content) <= 50:
                await self.status_emitter.emit(
                    f"❌ Failed to extract content from source {result.position}"
                )
                return None
            
            # Summarize content with quality rating
            summary_result = await content_summarizer.summarize_content(
                content, result.title, result.url, query, user_messages
            )
            
            if not summary_result or not summary_result.get("summary"):
                return None

            # Emit citation
            await self.citation_manager.emit_citation(
                result.title, result.url, summary_result.get("quality", 3), result.description, summary_result["summary"], message_id
            )
            
            return ProcessedSource(
                title=result.title,
                url=result.url,
                summary=summary_result["summary"],
                position=result.position or 0,
                published_date=result.published_date,
                quality=summary_result.get("quality", 3)
            )
        
        # Process all sources concurrently
        tasks = [process_single_source(result) for result in search_results]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        processed_sources = [result for result in results if result is not None]
        
        # Apply quality filtering logic
        return await self._filter_sources_by_quality(processed_sources)
    
    async def _filter_sources_by_quality(self, sources: List[ProcessedSource]) -> List[ProcessedSource]:
        """Filter sources by quality rating according to the specified rules."""
        if not sources:
            return sources
        
        # First, try to filter sources with quality > 2
        high_quality_sources = [source for source in sources if source.quality > 2]
        
        if high_quality_sources:
            await self.status_emitter.emit(
                f"🔍 Quality filtering: {len(high_quality_sources)}/{len(sources)} sources meet quality threshold (≥2)"
            )
            return high_quality_sources
        
        # If all sources are quality 1, or if all remaining sources are quality 3 or below,
        # check if we should include all sources
        max_quality = max(source.quality for source in sources)
        
        if max_quality <= 3:
            await self.status_emitter.emit(
                f"⚠️ All sources have quality ≤3 (max: {max_quality}) - including all sources as summarizer may not have understood quality criteria"
            )
            return sources
        
        # If we have sources with quality > 3, only return those with quality >= 2
        await self.status_emitter.emit(
            f"🔍 Quality filtering: {len(high_quality_sources)}/{len(sources)} sources meet quality threshold (≥2)"
        )
        return high_quality_sources
    
    def _create_error_response(self, search_results: List[SearchResult]) -> str:
        """Create error response when content extraction fails."""
        tavily_configured = "✅" if self.config.tavily_api_key else "❌"
        formatted_results = ResultFormatter.format_search_results(search_results)
        
        return (
            f"❌ **Content Extraction Failed**\n\n"
            f"**Search Results Found:** {len(search_results)}\n"
            f"**Possible Issues:**\n"
            f"- Tavily API key not configured: {tavily_configured}\n"
            f"- Websites may be blocking content extraction\n"
            f"- Content may be behind paywalls or require JavaScript\n\n"
            f"**Search Results Found:**\n{formatted_results}"
        )
    
    async def clear_cache(self) -> str:
        """Clear the internal cache."""
        self._cache.clear()
        return "✅ Cache cleared successfully."
    
    async def test_apis(self, event_emitter: Optional[EventEmitter] = None) -> str:
        """Test all configured APIs."""
        tester = ApiTester(self.config)
        return await tester.test_all_apis()


# --------------------------------------------------------------------------------------
# PUBLIC TOOL INTERFACE
# --------------------------------------------------------------------------------------
class Tools:
    """Main tool interface for Open WebUI."""

    class Valves(SearchConfig):
        """Configuration valves for the tool."""
        pass
    
    def __init__(self):
        self.valves = self.Valves()
        self.orchestrator = SmartSearchOrchestrator(self.valves)

    async def generate_citations(
        self,
        __metadata__: Optional[Dict[str, Any]] = None,
        __messages__: Optional[List[Dict[str, str]]] = None,
        __event_emitter__: Optional[Callable[[Any], None]] = None,
    ) -> List[Dict[str, str]]:
        """Generate citations for the recent searches. Use this before finalizing the assistant message if appropriate.
        Output format: [{"title": "<title>", "url": "<url>", "citation_str": "[<index>]"}]"""
        self.orchestrator.config = self.valves

        await self.orchestrator.status_emitter.emit("📚 Gathering citations...")

        # Get a list of previous assistant messages up to the most recent user message
        # This allows us to gather all the citations for the various turns of the assistant message
        # Get all assistant message IDs from the conversation up to the most recent user message
        assistant_message_ids = []
        if __messages__:
            # Start from the end and work backwards to find the most recent user message
            for i in range(len(__messages__) - 1, -1, -1):
                message = __messages__[i]
                if message.get("role") == "user":
                    # Found the most recent user message, now collect all assistant messages before it
                    for j in range(i):
                        prev_message = __messages__[j]
                        if prev_message.get("role") == "assistant":
                            # Extract message ID from metadata if available
                            msg_id = prev_message.get("id") or prev_message.get("message_id")
                            if msg_id:
                                assistant_message_ids.append(msg_id)
                    break

        await asyncio.sleep(0.1)  # Allow event loop to process
        if self.orchestrator.citation_manager:
            message_id = __metadata__.get("message_id") if __metadata__ else None
            raw_citations = self.orchestrator.citation_manager.gather_citations(message_id) if message_id else []

            # Get citations to previous turns of the assistant message
            for msg_id in assistant_message_ids:
                raw_citations.extend(self.orchestrator.citation_manager.gather_citations(msg_id))
        
            print(f"All citations: {self.orchestrator.citation_manager.citations}")
            print(f"Raw citations: {raw_citations}")
            # Change format to be understood by LLMs "[{"title": "<title>", "url": "<url>", "citation_str": "[<index>]"}]"
            citations = []
            for i, citation in enumerate(raw_citations, 1):
                citations.append({
                    "title": citation["source"]["name"],
                    "url": citation["source"]["url"],
                    "citation_str": f"【{i}】"
                })

            # Clear citations
            self.orchestrator.citation_manager.citations = {}
            
            message_id = __metadata__.get("message_id") if __metadata__ else None
            return {"message_id": message_id, "citations": citations}
        
        return [{"title": "No Citations", "url": "", "citation_str": ""}]

    async def search_web(
        self,
        query: str,
        allow_split: bool = True,
        __event_emitter__: Optional[Callable[[Any], None]] = None,
        __user__: Optional[Dict] = None,
        __messages__: Optional[List[Dict[str, str]]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Perform a web search that returns a detailed summary.

        params: query (str): The search query.
                allow_split (bool): Whether to allow splitting the query into multiple parts. Setting to false will allow for more targeted queries. (Default: True)
        """
        self.orchestrator.config = self.valves

        message_id = __metadata__.get("message_id") if __metadata__ else None
        #print(f"[DEBUG] search_web: __metadata__ = {__metadata__}, message_id = {repr(message_id)}")
        return await self.orchestrator.search_and_summarize(
            query, allow_split, __event_emitter__, __messages__, message_id
        )

    async def read_web_page(
            self,
            url: str,
            summarize: bool = True,
            __event_emitter__: Optional[Callable[[Any], None]] = None,
            __user__: Optional[Dict] = None,
            __messages__: Optional[List[Dict[str, str]]] = None,
            __metadata__: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Read a web page. Can either return the raw content or a summary."""
        self.orchestrator.config = self.valves

        # Read the most recent message
        msg = __messages__[-1]["content"] if __messages__ else ""
        message_id = __metadata__.get("message_id") if __metadata__ else None
        print(f"[DEBUG] read_web_page: __metadata__ = {__metadata__}, message_id = {repr(message_id)}")
        return await self.orchestrator.read_web_page(url, msg, summarize, __event_emitter__, __messages__, message_id)
