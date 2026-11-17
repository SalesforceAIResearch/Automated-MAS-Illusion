"""Simple-MAS prompt agents; shared meta schema with Expert-MAS."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
from evaluation.expert_MAS_for_smfr_dataset.agents import MetaAgent, PipelineStats, CallUsage, ProblemMetaResponse
from evaluation.common.providers import PipelineClient as SimpleMASClient
PROMPTS_DIR = Path(__file__).parent / "prompts"
class RetrievalAgent:
    """
    One LLM call: given the full problem text and investor names, extract every
    transaction for every investor along with the exact transaction-date price.
    Returns raw text (the LLM's response) and CallUsage.
    No Pydantic enforcement — output is parsed from free-form JSON.
    """

    def __init__(self, model_client: SimpleMASClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "retrieval.txt").read_text()

    async def extract(
        self,
        problem_text: str,
        investors: list,
        price_type: str,
    ) -> Tuple[Optional[str], CallUsage]:
        investors_str = ", ".join(investors)
        prompt = (
            self._template
            .replace("{problem_text}", problem_text)
            .replace("{investors}", investors_str)
            .replace("{price_type}", price_type)
        )
        return await self.model.generate_text(prompt)

class CalculationAgent:
    """
    One LLM call: receives extracted transaction data + full problem text (for
    price table scanning) + meta parameters. Returns the final answer dict as
    free-form JSON: {"investor_dates": {...}, "answer": [...]}.
    """

    def __init__(self, model_client: SimpleMASClient):
        self.model = model_client
        self._template = (PROMPTS_DIR / "calculation.txt").read_text()

    async def calculate(
        self,
        problem_text: str,
        extracted_data: str,
        meta: ProblemMetaResponse,
    ) -> Tuple[Optional[str], CallUsage]:
        prompt = (
            self._template
            .replace("{problem_text}", problem_text)
            .replace("{extracted_data}", extracted_data)
            .replace("{question_type}", meta.question_type)
            .replace("{target_pct}", str(meta.target_percentage))
            .replace("{price_type}", meta.price_type)
            .replace("{aggregation}", meta.aggregation)
        )
        return await self.model.generate_text(prompt)
