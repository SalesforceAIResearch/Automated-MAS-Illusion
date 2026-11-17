"""The corrected GPT-OSS representation adapter; financial logic is unchanged."""
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict
from evaluation.common.providers import PipelineClient, strict_schema


class Transaction(BaseModel):
    model_config = ConfigDict(extra='forbid')
    company: str
    date: str
    action: Literal['buy', 'sell']
    quantity: int
    price: float


class TransactionsResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    result: list[Transaction]
    reasoning: str


class FinancialTarget(BaseModel):
    model_config = ConfigDict(extra='forbid')
    total_cost: float
    realized_pnl: float
    target_profit: float
    profit_needed: float
    required_price: float


class CalculationResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    result: FinancialTarget
    reasoning: str


class DatesResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    result: list[str]
    reasoning: str


def stage_model(prompt, response_format):
    if response_format.__name__ == 'ProblemMetaResponse':
        return response_format
    if response_format.__name__ == 'CalculateResponse':
        return CalculationResponse
    if response_format.__name__ == 'ExtractResponse':
        if 'From the transaction events section ONLY' in prompt:
            return TransactionsResponse
        if 'stock price history in the context and list ALL dates' in prompt:
            return DatesResponse
    raise ValueError('Unrecognized Expert-MAS stage; refuse to guess a schema')


class TypedExpertClient(PipelineClient):
    async def generate(self, prompt, response_format):
        typed_model = stage_model(prompt, response_format)
        schema = strict_schema(typed_model.model_json_schema())
        fields = list(response_format.model_json_schema().get('properties', {}))
        prompt += f'\n\nRespond with a JSON object containing exactly these fields: {fields}. Output only valid JSON, no markdown fences.'
        if typed_model is not response_format:
            description = {
                TransactionsResponse: 'a JSON array of transaction objects',
                CalculationResponse: 'a JSON object containing the financial target fields',
                DatesResponse: 'a JSON array of date strings',
            }[typed_model]
            prompt += f'\n\nFor this request, the output schema below overrides earlier serialization examples: result must be {description}, not a JSON-encoded string. Follow this schema: ' + json.dumps(schema)
        typed, usage = await super().generate(prompt, typed_model)
        if typed is None or typed_model is response_format:
            return typed, usage
        return response_format(result=json.dumps(typed.model_dump()['result']), reasoning=typed.reasoning), usage
