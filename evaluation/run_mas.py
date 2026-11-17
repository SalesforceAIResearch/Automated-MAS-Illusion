"""Run one checkpointed repetition of our Simple-MAS or Expert-MAS pipeline."""
import argparse
import asyncio
import contextvars
import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time

from dataset.evaluate.scoring import compare_answers
from evaluation.common.data import ROOT, dataset, digest, save, write_jsonl
from evaluation.common.providers import CALL_CONTEXT, PipelineClient, Provider, now
from evaluation.expert_MAS_for_smfr_dataset.client import TypedExpertClient


def instrument(pipeline):
    """Record actual investor/position dependencies without changing prompts."""
    for name, stage in [('_extract_transactions_with_prices', 'retrieval'),
                        ('_calculate_pnl_and_target', 'calculation'), ('_extract_valid_dates', 'dates')]:
        original = getattr(pipeline, name)
        if getattr(original, '_smfr_instrumented', False):
            continue
        signature = inspect.signature(original)
        def wrapper(original, signature, stage):
            @functools.wraps(original)
            async def run(*args, **kwargs):
                values = signature.bind(*args, **kwargs).arguments
                context = dict(CALL_CONTEXT.get())
                context['stage'] = stage
                if stage == 'retrieval':
                    context.update(investor=values['investor'], branch=values['investor'])
                elif stage == 'calculation':
                    context.update(investor=values['investor'],
                                   branch=json.dumps([values['investor'], values['open_position']], sort_keys=True))
                CALL_CONTEXT.set(context)
                return await original(*args, **kwargs)
            run._smfr_instrumented = True
            return run
        setattr(pipeline, name, wrapper(original, signature, stage))


async def run(args):
    module = importlib.import_module(f'evaluation.{args.system}_MAS_for_smfr_dataset.pipeline')
    if args.system == 'expert':
        instrument(module)
    rows = dataset(args.investors, args.input)
    if args.limit:
        rows = rows[:args.limit]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    is_oss = 'gpt-oss' in args.model
    is_gemini = 'gemini' in args.model
    temperature = 0.0 if is_gemini or not any(x in args.model for x in ['gpt-5', 'gpt-oss', 'o3', 'o4']) else None
    max_tokens = args.max_output_tokens if args.max_output_tokens is not None else (32768 if is_oss else 65536 if is_gemini and args.system == 'simple' else None)
    reasoning = 'medium' if is_oss else None
    package = ROOT / f'evaluation/{args.system}_MAS_for_smfr_dataset'
    source_files = [package / 'pipeline.py', package / 'agents.py', *sorted((package / 'prompts').glob('*.txt'))]
    manifest = dict(system=args.system, model=args.model, provider=args.provider, run_id=args.run_id,
                    selected_problem_sha256=[digest(r['problem']) for _, r in rows],
                    temperature=temperature, max_output_tokens=max_tokens, reasoning_effort=reasoning,
                    expert_schema='typed_gptoss' if is_oss and args.system == 'expert' else 'original',
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                    problem_concurrency=args.workers, call_concurrency=args.max_calls,
                    automatic_followup_runs=False, code_version='public-release-v1')
    previous = output / 'manifest.json'
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise RuntimeError('Run configuration changed; choose a different output directory')
    save(previous, manifest)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, cases=len(rows), manifest=str(previous)), indent=2))
        return
    provider = Provider(args.provider, args.model, args.max_calls, args.input_rate, args.output_rate)
    semaphore = asyncio.Semaphore(args.workers)
    errors = []
    stop = asyncio.Event()
    async def evaluate(n, sample):
        key = digest(sample['problem'])
        result_path = output / 'results' / f'n{n}_{key}.json'
        async with semaphore:
            if result_path.exists() or stop.is_set() or (output / 'STOP').exists():
                return
            cls = TypedExpertClient if args.system == 'expert' and is_oss else PipelineClient
            client = cls(provider, output / 'cases' / f'n{n}_{key}', temperature=temperature,
                         max_tokens=max_tokens, reasoning_effort=reasoning,
                         prompt_only_schema=is_oss and args.system == 'simple',
                         raw_json=is_gemini and args.system == 'simple')
            CALL_CONTEXT.set(dict(stage='meta'))
            started = time.monotonic()
            try:
                answer, stats, trace = await module.run_pipeline(sample['problem'], client)
            except (KeyError, ValueError, TypeError) as error:
                # An invalid returned payload is an outcome, never a resampling gate.
                save(result_path, dict(input=sample, output={}, status='model_payload_failure',
                                       error=type(error).__name__, correct=False, completed_utc=now()))
                return
            except Exception as error:
                errors.append(dict(problem_sha256=key, exception=type(error).__name__))
                save(output / 'errors' / f'{key}.json', errors[-1])
                stop.set()
                return
            save(result_path, dict(input=sample, model=args.model, output=dict(answer=answer),
                                   stats=stats.to_dict(), trace=trace, correct=compare_answers(sample['answer'], dict(answer=answer)),
                                   wall_seconds=time.monotonic() - started, completed_utc=now(), run_id=args.run_id))
            print(json.dumps(dict(event='case_finished', case=key, completed=len(list((output / 'results').glob('*.json'))))), flush=True)
    await asyncio.gather(*(evaluate(n, r) for n, r in rows))
    results = [json.loads(p.read_text()) for p in sorted((output / 'results').glob('*.json'))]
    usage = [json.loads(p.read_text()) for p in output.glob('cases/*/calls/*/usage.json')]
    costed = [r for r in usage if r['cost_usd'] is not None]
    summary = dict(target_cases=len(rows), completed_cases=len(results), correct=sum(r['correct'] for r in results),
                   accuracy_pct=100 * sum(r['correct'] for r in results) / len(rows) if len(results) == len(rows) else None,
                   calls=len(usage), input_tokens=sum(r['input_tokens'] for r in usage), output_tokens=sum(r['output_tokens'] for r in usage),
                   recorded_cost_usd=sum(r['cost_usd'] for r in costed) if len(costed) == len(usage) and usage else None,
                   costed_calls=len(costed), errors=errors, completed_utc=now())
    save(output / 'summary.json', summary)
    save(output / ('complete.json' if len(results) == len(rows) else 'incomplete.json'), summary)
    write_jsonl(output / 'outputs.jsonl', results)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=['simple', 'expert'], required=True)
    parser.add_argument('--provider', choices=['openai', 'together', 'openrouter', 'vertex'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--investors', nargs='+', type=int, choices=range(2, 7))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-calls', type=int, default=16)
    parser.add_argument('--max-output-tokens', type=int)
    parser.add_argument('--run-id', type=int, default=0)
    parser.add_argument('--input-rate', type=float, help='Optional USD per million input tokens')
    parser.add_argument('--output-rate', type=float, help='Optional USD per million output tokens, including reasoning')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.max_calls < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Concurrency and limit must be positive')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
