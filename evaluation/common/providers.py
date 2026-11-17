"""Public provider adapters with raw responses, per-call usage, and bounded retries.

No credentials are loaded until a provider is explicitly instantiated. There is
no built-in organization gateway, project, account, or endpoint selection.
"""
import asyncio
from collections import Counter
import contextvars
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from .answers import parse_response
from .data import save

CALL_CONTEXT = contextvars.ContextVar('smfr_call_context', default={})


def now():
    return datetime.now(timezone.utc).isoformat()


def strict_schema(value):
    if isinstance(value, dict):
        result = {k: strict_schema(v) for k, v in value.items()}
        if result.get('type') == 'object':
            result['additionalProperties'] = False
            result['required'] = list(result.get('properties', {}))
        return result
    if isinstance(value, list):
        return [strict_schema(v) for v in value]
    return value


def vertex_usage(raw):
    usage = raw.get('usage_metadata') or {}
    if usage.get('prompt_token_count') is None:
        raise RuntimeError('Cached response has no token usage; inspect without resampling')
    prompt = usage['prompt_token_count']
    thinking = usage.get('thoughts_token_count') or 0
    output = max((usage.get('candidates_token_count') or 0) + thinking,
                 (usage.get('total_token_count') or prompt) - prompt)
    return dict(input_tokens=prompt, output_tokens=output, thinking_tokens=thinking, raw_usage=usage)


class Provider:
    def __init__(self, name, model, call_limit=32, input_rate=None, output_rate=None):
        self.name, self.model = name, model
        self.semaphore = asyncio.Semaphore(call_limit)
        self.input_rate, self.output_rate = input_rate, output_rate
        if name == 'vertex':
            from google import genai
            from google.genai import types
            project = os.environ.get('GOOGLE_CLOUD_PROJECT')
            if not project:
                raise EnvironmentError('Set GOOGLE_CLOUD_PROJECT and authenticate Application Default Credentials')
            self.client = genai.Client(vertexai=True, project=project,
                                       location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'),
                                       http_options=types.HttpOptions(timeout=1200000,
                                           retry_options=types.HttpRetryOptions(attempts=1)))
        else:
            from openai import OpenAI
            if name == 'together':
                key, url = os.environ.get('TOGETHER_API_KEY'), 'https://api.together.ai/v1'
            elif name == 'openrouter':
                key, url = os.environ.get('OPENROUTER_API_KEY'), 'https://openrouter.ai/api/v1'
            else:
                key = os.environ.get('OPENAI_API_KEY')
                url = os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
            if not key:
                raise EnvironmentError(f'API key is not configured for {name}')
            self.client = OpenAI(api_key=key, base_url=url, max_retries=0, timeout=1200)

    async def complete(self, messages, directory, *, temperature=None, max_tokens=None,
                       reasoning_effort=None, seed=None, schema=None, json_mode=False, metadata=None, verbosity=None):
        options = dict(model=self.model, messages=messages)
        for key, value in [('temperature', temperature), ('max_tokens', max_tokens),
                           ('reasoning_effort', reasoning_effort), ('seed', seed), ('verbosity', verbosity)]:
            if value is not None:
                options[key] = value
        if schema is not None:
            options['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': schema.get('title', 'Response'), 'strict': True,
                'schema': schema if self.name == 'vertex' else strict_schema(schema)}}
        elif json_mode:
            options['response_format'] = {'type': 'json_object'}
        if self.name == 'openrouter':
            options['extra_body'] = {'provider': {'require_parameters': True}}
        request = dict(provider=self.name, options=options)
        async with self.semaphore:
            return await asyncio.to_thread(self._complete, request, Path(directory), metadata or {})

    def _complete(self, request, directory, metadata):
        import httpx
        request_path = directory / 'request.json'
        if request_path.exists() and json.loads(request_path.read_text()) != request:
            raise RuntimeError('Checkpoint request changed; use a new output directory')
        save(request_path, request)
        options = request['options']
        raw_path = directory / 'response.json'
        started = time.monotonic()
        if raw_path.exists():
            raw = json.loads(raw_path.read_text())
        else:
            for attempt in range(6):
                attempt_path = directory / 'attempts' / f'{time.time_ns()}.json'
                entry = dict(started_utc=now(), status='inflight')
                save(attempt_path, entry)
                try:
                    if self.name == 'vertex':
                        from google.genai import types
                        config = {}
                        if options.get('temperature') is not None:
                            config['temperature'] = options['temperature']
                        if options.get('max_tokens') is not None:
                            config['max_output_tokens'] = options['max_tokens']
                        if options.get('seed') is not None:
                            config['seed'] = options['seed']
                        messages = options['messages']
                        systems = [m['content'] for m in messages if m['role'] == 'system']
                        if systems:
                            config['system_instruction'] = '\n\n'.join(systems)
                        contents = [dict(role='model' if m['role'] == 'assistant' else 'user',
                                         parts=[dict(text=m['content'])]) for m in messages if m['role'] != 'system']
                        fmt = options.get('response_format', {})
                        if fmt:
                            config['response_mime_type'] = 'application/json'
                            if fmt['type'] == 'json_schema':
                                config['response_schema'] = fmt['json_schema']['schema']
                        # Vertex request schemas preserve the supplied stage schema;
                        # strict OpenAI conversion above is semantically equivalent.
                        response = self.client.models.generate_content(model=self.model, contents=contents,
                                                                       config=types.GenerateContentConfig(**config))
                    else:
                        outgoing = dict(options)
                        if 'gpt-5' in self.model or self.model.startswith(('o3', 'o4')):
                            if outgoing.get('max_tokens') is not None:
                                outgoing['max_completion_tokens'] = outgoing.pop('max_tokens')
                        response = self.client.chat.completions.create(**outgoing)
                    raw = response.model_dump(mode='json')
                    save(raw_path, raw)  # Always checkpoint before parsing or local execution.
                    save(attempt_path, {**entry, 'status': 'response_saved', 'completed_utc': now()})
                    break
                except Exception as error:
                    code = getattr(error, 'status_code', None) or getattr(error, 'code', None)
                    transient = code in {408, 429, 500, 502, 503, 504} or isinstance(error, httpx.TransportError)
                    if type(error).__name__ in {'APIConnectionError', 'APITimeoutError'}:
                        transient = True
                    save(attempt_path, {**entry, 'status': 'failed', 'exception': type(error).__name__,
                         'http_status': code, 'billing_unknown': code is None or code in {408, 500, 502, 503, 504}})
                    if not transient or attempt == 5:
                        raise
                    time.sleep(min(120, 15 * 2 ** attempt))
        if self.name == 'vertex':
            usage = vertex_usage(raw)
            candidates = raw.get('candidates') or []
            parts = ((candidates[0].get('content') or {}).get('parts') or []) if candidates else []
            text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
            finish = candidates[0].get('finish_reason') if candidates else 'NO_CANDIDATE'
        else:
            recorded = raw.get('usage') or {}
            if recorded.get('prompt_tokens') is None or recorded.get('completion_tokens') is None:
                raise RuntimeError('Cached response has no token usage; inspect without resampling')
            usage = dict(input_tokens=recorded['prompt_tokens'], output_tokens=recorded['completion_tokens'],
                         thinking_tokens=(recorded.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                         raw_usage=recorded)
            choices = raw.get('choices') or []
            message = (choices[0].get('message') or choices[0].get('delta') or {}) if choices else {}
            text = message.get('content') or ''
            finish = choices[0].get('finish_reason') if choices else 'NO_CHOICE'
        cost = (raw.get('usage') or {}).get('cost')
        cost_source = 'api_reported' if cost is not None else None
        if cost is None and self.input_rate is not None and self.output_rate is not None:
            cost = (usage['input_tokens'] * self.input_rate + usage['output_tokens'] * self.output_rate) / 1e6
            cost_source = 'user_supplied_rates'
        usage.update(cost_usd=cost, cost_source=cost_source, provider=self.name, model=self.model,
                     finish_reason=finish, metadata=metadata)
        save(directory / 'usage.json', usage)
        if not (directory / 'timing.json').exists():
            save(directory / 'timing.json', dict(elapsed_seconds=time.monotonic() - started, completed_utc=now()))
        return text, usage


class PipelineClient:
    """Meta/raw-text interface used by our unmodified MAS pipelines."""
    def __init__(self, provider, directory, *, temperature=None, max_tokens=None,
                 reasoning_effort=None, prompt_only_schema=False, raw_json=False):
        self.provider, self.directory = provider, Path(directory)
        self.model_name = provider.model.rsplit('/', 1)[-1]
        self.temperature, self.max_tokens = temperature, max_tokens
        self.reasoning_effort = reasoning_effort
        self.prompt_only_schema, self.raw_json = prompt_only_schema, raw_json
        self.occurrences = Counter()
        self.raw_index = 0

    async def call(self, prompt, response_format=None):
        schema = response_format.model_json_schema() if response_format else None
        if self.prompt_only_schema and schema:
            fields = list(schema.get('properties', {}))
            prompt += f'\n\nRespond with a JSON object containing exactly these fields: {fields}. Output only valid JSON, no markdown fences.'
            schema = None
        options = dict(prompt=prompt, schema=schema, temperature=self.temperature,
                       max_tokens=self.max_tokens, reasoning_effort=self.reasoning_effort)
        fingerprint = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
        index = self.occurrences[fingerprint]
        self.occurrences[fingerprint] += 1
        context = dict(CALL_CONTEXT.get())
        if response_format is not None and response_format.__name__ == 'ProblemMetaResponse':
            context = dict(stage='meta')
        elif response_format is None:
            context = dict(stage=['retrieval', 'calculation'][self.raw_index])
            self.raw_index += 1
        call_name = (response_format.__name__ if response_format else context.get('stage', 'text'))
        text, usage = await self.provider.complete([dict(role='user', content=prompt)],
            self.directory / 'calls' / f'{call_name}_{fingerprint}_{index}', temperature=self.temperature,
            max_tokens=self.max_tokens, reasoning_effort=self.reasoning_effort,
            schema=schema, json_mode=self.raw_json and response_format is None, metadata=context)
        from evaluation.expert_MAS_for_smfr_dataset.agents import CallUsage
        return text, CallUsage(input_tokens=usage['input_tokens'], output_tokens=usage['output_tokens'])

    async def generate(self, prompt, response_format):
        text, usage = await self.call(prompt, response_format)
        try:
            return response_format.model_validate(parse_response(text)), usage
        except (ValueError, TypeError):
            return None, usage

    async def generate_text(self, prompt):
        return await self.call(prompt)
