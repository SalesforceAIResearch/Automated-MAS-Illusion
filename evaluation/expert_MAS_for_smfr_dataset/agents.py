"""Expert-MAS schemas, prompt agents, and token accounting."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Type
from pydantic import BaseModel
PROMPTS_DIR = Path(__file__).parent / "prompts"

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":              {"input": 2.50,  "output": 10.00},
    "gpt-4o-2024-08-06":   {"input": 2.50,  "output": 10.00},
    "gpt-4.1":             {"input": 2.00,  "output":  8.00},
    "gpt-4.1-2025-04-14":  {"input": 2.00,  "output":  8.00},
    "o3":                  {"input": 10.00, "output": 40.00},
    "o3-2025-04-16":       {"input": 10.00, "output": 40.00},
    "o4-mini":             {"input": 1.10,  "output":  4.40},
    "o4-mini-2025-04-16":  {"input": 1.10,  "output":  4.40},
    "gpt-5":               {"input": 1.25,  "output": 10.00},
    "gpt-5-2025-08-07":    {"input": 1.25,  "output": 10.00},
    # Google Gemini
    "gemini-2.0-flash":               {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro":                 {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash":               {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite-preview-06-17": {"input": 0.10, "output": 0.40},
    "gemini-3-pro-preview":           {"input": 2.00, "output": 12.00},
    # Open-source models (via Together.ai-compatible gateway)
    "gpt-oss-120b":                   {"input": 0.15,  "output":  0.60},  # Together.ai pricing
    # Anthropic Claude
    "claude-haiku-4-5":               {"input": 0.80,  "output":  4.00},
    "claude-haiku-4-5-20251001":      {"input": 0.80,  "output":  4.00},
    "claude-sonnet-4-6":              {"input": 3.00,  "output": 15.00},
    "claude-opus-4-6":                {"input": 15.00, "output": 75.00},
}
def get_pricing(model_name: str) -> Optional[Dict[str, float]]:
    """Look up pricing by exact match, then by prefix (handles versioned IDs)."""
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    for key in MODEL_PRICING:
        if model_name.startswith(key):
            return MODEL_PRICING[key]
    return None

@dataclass
class CallUsage:
    """Token usage for a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

@dataclass
class PipelineStats:
    """Accumulated token usage and cost across all LLM calls in one pipeline run."""
    model_name: str = ""
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Per-phase breakdowns (phase name → CallUsage)
    phases: Dict[str, CallUsage] = field(default_factory=dict)

    def add(self, usage: CallUsage, phase: str = ""):
        """Record usage from one LLM call."""
        self.llm_calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        if phase:
            if phase not in self.phases:
                self.phases[phase] = CallUsage()
            self.phases[phase].input_tokens += usage.input_tokens
            self.phases[phase].output_tokens += usage.output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self) -> Dict[str, float]:
        """Return {"input": $, "output": $, "total": $} for this run."""
        pricing = get_pricing(self.model_name)
        if pricing is None:
            return {"input": 0.0, "output": 0.0, "total": 0.0, "pricing_found": False}
        input_cost = self.input_tokens / 1_000_000 * pricing["input"]
        output_cost = self.output_tokens / 1_000_000 * pricing["output"]
        return {
            "input": round(input_cost, 6),
            "output": round(output_cost, 6),
            "total": round(input_cost + output_cost, 6),
            "pricing_found": True,
        }

    def to_dict(self) -> Dict:
        cost = self.cost()
        return {
            "model": self.model_name,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": cost,
            "phases": {
                phase: {
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                }
                for phase, u in self.phases.items()
            },
        }

class ProblemMetaResponse(BaseModel):
    investors: List[str]
    question_type: str        # "sell" or "buy"
    target_percentage: float  # e.g. 1.3
    aggregation: str          # "earliest" or "latest"
    price_type: str           # "Open" or "Close"

class ExtractResponse(BaseModel):
    result: str    # JSON-encoded value — parse with json.loads()
    reasoning: str

class CalculateResponse(BaseModel):
    result: str    # JSON-encoded value — parse with json.loads()
    reasoning: str

class ModelClient(ABC):
    """Abstract base for all model backends.

    generate() returns (parsed_response, CallUsage).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def generate(
        self, prompt: str, response_format: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], CallUsage]:
        ...

class MetaAgent:
    """Parses raw problem text into structured parameters."""

    def __init__(self, model_client: ModelClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "meta_agent.txt").read_text()

    async def parse(
        self, problem_text: str
    ) -> Tuple[Optional[ProblemMetaResponse], CallUsage]:
        prompt = self._template.replace("{problem_text}", problem_text)
        return await self.model.generate(prompt, ProblemMetaResponse)

class ExtractAgent:
    """Generic information retrieval: extract(context, query) → (result, usage)."""

    def __init__(self, model_client: ModelClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "extract.txt").read_text()

    async def extract(
        self, context: str, query: str
    ) -> Tuple[Optional[ExtractResponse], CallUsage]:
        prompt = self._template.replace("{context}", context).replace("{query}", query)
        return await self.model.generate(prompt, ExtractResponse)

class CalculateAgent:
    """Generic numerical computation: calculate(data, query) → (result, usage)."""

    def __init__(self, model_client: ModelClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "calculate.txt").read_text()

    async def calculate(
        self, data: str, query: str
    ) -> Tuple[Optional[CalculateResponse], CallUsage]:
        prompt = self._template.replace("{data}", data).replace("{query}", query)
        return await self.model.generate(prompt, CalculateResponse)
