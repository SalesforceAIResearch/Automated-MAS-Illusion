"""Standalone CoT/CoT-SC and Code-CoT/Code-CoT-SC, without MAS dependencies."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluation.common.answers import answer_key, parse_response, parsed_answer, vote_keys
from evaluation.common.data import ROOT, dataset, digest, save, write_jsonl
from evaluation.common.execution import execute
from evaluation.common.providers import Provider, now

PROMPTS = Path(__file__).parent / 'prompts'


def messages_for(protocol, problem):
    # Replace the literal sentinel only; braces within problems/code stay literal.
    return [dict(role=m['role'], content=m['content'].replace('{{problem}}', problem.strip()))
            for m in protocol['messages']]


def selected_vote(draws, policy):
    if policy == 'canonical_name_sets':
        return vote_keys([parsed_answer(row) for row in draws])
    values = [row['answer'] for row in draws if isinstance(row, dict) and 'answer' in row]
    if policy == 'raw_answer':
        values = [v if isinstance(v, str) else json.dumps(v, sort_keys=True) for v in values]
        return answer_key(Counter(values).most_common(1)[0][0]) if values else None
    # Historical GPT-OSS direct voting uses case-sensitive name multisets and
    # excludes None / empty / string "None" before voting. The emitted answer
    # is subsequently canonicalized for comparison with the canonical labels.
    candidates = []
    import ast
    import re
    for value in values:
        if value is None or (isinstance(value, str) and value.strip().lower() in {'', 'none'}):
            continue
        if isinstance(value, str):
            text = value.strip()
            if text.startswith('['):
                try:
                    value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    value = text
            if isinstance(value, str):
                value = re.split(r'\s+and\s+|,', value)
        names = value if isinstance(value, list) else [value]
        candidates.append(tuple(sorted(str(x).strip() for x in names if str(x).strip())))
    winner = Counter(candidates).most_common(1)[0][0] if candidates else None
    return answer_key(list(winner)) if winner is not None else None


async def run(args):
    protocol_path = PROMPTS / f'{args.protocol}.json'
    protocol = json.loads(protocol_path.read_text())
    rows = dataset(args.investors, args.input)
    if args.limit:
        rows = rows[:args.limit]
    k = args.k if args.k is not None else protocol['default_k']
    is_oss, is_gpt5 = 'gpt-oss' in args.model, 'gpt-5' in args.model
    temperature = protocol.get('temperature')
    max_tokens = protocol.get('max_output_tokens')
    effort = protocol.get('reasoning_effort')
    verbosity = None
    base_seed = args.seed
    if args.protocol.startswith('controlled'):
        max_tokens = 32768 if is_gpt5 or is_oss else 16384
        temperature = None if is_gpt5 or is_oss else .8
        effort = 'medium' if is_gpt5 or is_oss else None
        verbosity = 'low' if is_gpt5 else None
        if base_seed is None:
            base_seed = args.run_id * 100
    if args.max_output_tokens is not None:
        max_tokens = args.max_output_tokens
    args.output.mkdir(parents=True, exist_ok=True)
    method = ('Code-CoT' if protocol['code_requested'] else 'CoT') + ('-SC' if k > 1 else '')
    manifest = dict(system=method,
                    model=args.model, provider=args.provider, protocol=args.protocol, k=k, run_id=args.run_id,
                    temperature=temperature, max_output_tokens=max_tokens, reasoning_effort=effort,
                    seed_policy='omitted' if base_seed is None else 'base + draw index', base_seed=base_seed,
                    verbosity=verbosity,
                    execute_code=args.execute_code, prompt_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                    cases=[digest(r['problem']) for _, r in rows], workers=args.workers,
                    voting=protocol['voting'], automatic_followup_runs=False)
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Run configuration changed; choose another output directory')
    save(path, manifest)
    if args.dry_run:
        save(args.output / 'example_request.json', dict(messages=messages_for(protocol, rows[0][1]['problem']),
                                                       temperature=temperature, max_tokens=max_tokens, reasoning_effort=effort))
        print(json.dumps(dict(dry_run=True, problems=len(rows), calls=len(rows) * k, manifest=str(path)), indent=2))
        return
    provider = Provider(args.provider, args.model, args.workers, args.input_rate, args.output_rate)
    queue = asyncio.Queue()
    for n, row in rows:
        for index in range(k):
            queue.put_nowait((n, row, index))
    errors = []
    stop = asyncio.Event()
    async def worker():
        while not queue.empty() and not stop.is_set() and not (args.output / 'STOP').exists():
            n, row, index = queue.get_nowait()
            key = digest(row['problem'])
            directory = args.output / 'cases' / f'n{n}_{key}' / f'draw_{index}'
            if (directory / 'parsed.json').exists():
                continue
            try:
                text, usage = await provider.complete(messages_for(protocol, row['problem']), directory,
                    temperature=temperature, max_tokens=max_tokens, reasoning_effort=effort,
                    seed=None if base_seed is None else base_seed + index, verbosity=verbosity,
                    schema=protocol.get('schema'),
                    json_mode=protocol.get('json_mode', False) and not (args.protocol.startswith('controlled') and is_gpt5),
                    metadata=dict(problem_sha256=key, investors=n, draw=index))
                parsed = parse_response(text)
                save(directory / 'parsed.json', dict(parsed=parsed, finish_reason=usage['finish_reason']))
                if args.execute_code:
                    execution = await asyncio.to_thread(execute, parsed.get('code') if isinstance(parsed, dict) else None, args.timeout)
                    save(directory / 'execution.json', execution)
                print(json.dumps(dict(event='draw_finished', problem_sha256=key, draw=index)), flush=True)
            except Exception as error:
                errors.append(dict(problem_sha256=key, draw=index, exception=type(error).__name__))
                save(directory / 'error.json', errors[-1])
                stop.set()
    await asyncio.gather(*(worker() for _ in range(args.workers)))
    results = []
    for n, row in rows:
        key = digest(row['problem'])
        directories = [args.output / 'cases' / f'n{n}_{key}' / f'draw_{i}' for i in range(k)]
        if not all((d / 'parsed.json').exists() for d in directories):
            continue
        draws = [json.loads((d / 'parsed.json').read_text())['parsed'] for d in directories]
        text_key = selected_vote(draws, protocol['voting'])
        execution_key = None
        if args.execute_code:
            executions = []
            for directory, parsed in zip(directories, draws):
                path = directory / 'execution.json'
                if not path.exists():
                    save(path, execute(parsed.get('code') if isinstance(parsed, dict) else None, args.timeout))
                executions.append(json.loads(path.read_text()))
            execution_key = vote_keys([parsed_answer(x.get('result')) if x.get('success') else None for x in executions])
        result = dict(input=row, problem_sha256=key, investors=n, k=k,
                      output=dict(answer=list(text_key)) if text_key is not None else {},
                      text_correct=text_key is not None and text_key == answer_key(row['answer']),
                      execution_correct=execution_key is not None and execution_key == answer_key(row['answer']),
                      execution_vote=list(execution_key) if execution_key is not None else None,
                      draws=draws, completed_utc=now())
        save(args.output / 'results' / f'n{n}_{key}.json', result)
        results.append(result)
    usage = [json.loads(p.read_text()) for p in args.output.glob('cases/*/draw_*/usage.json')]
    costed = [u for u in usage if u['cost_usd'] is not None]
    summary = dict(target_cases=len(rows), completed_cases=len(results), k=k, calls=len(usage),
                   text_correct=sum(r['text_correct'] for r in results), execution_correct=sum(r['execution_correct'] for r in results),
                   accuracy_pct=100 * sum(r['text_correct'] for r in results) / len(rows) if len(rows) == len(results) else None,
                   input_tokens=sum(u['input_tokens'] for u in usage), output_tokens=sum(u['output_tokens'] for u in usage),
                   recorded_cost_usd=sum(u['cost_usd'] for u in costed) if len(costed) == len(usage) and usage else None,
                   costed_calls=len(costed), errors=errors, completed_utc=now())
    save(args.output / 'summary.json', summary)
    save(args.output / ('complete.json' if len(results) == len(rows) else 'incomplete.json'), summary)
    write_jsonl(args.output / 'outputs.jsonl', results)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=sorted(p.stem for p in PROMPTS.glob('*.json')), required=True)
    parser.add_argument('--provider', choices=['openai', 'together', 'openrouter', 'vertex'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--investors', type=int, nargs='+', choices=range(2, 7))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--run-id', type=int, default=0)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-output-tokens', type=int)
    parser.add_argument('--execute-code', action='store_true')
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--input-rate', type=float)
    parser.add_argument('--output-rate', type=float)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.k is not None and args.k < 1) or (args.limit is not None and args.limit < 1):
        parser.error('Workers, k, and limit must be positive')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
