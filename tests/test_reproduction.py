"""Offline checks: no provider credentials and no paid model calls."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import argparse
from unittest.mock import patch

from dataset.evaluate.scoring import compare_answers
from evaluation.common.answers import answer_key, parsed_answer, vote_keys
from evaluation.common.execution import execute
from evaluation.common.providers import PipelineClient, Provider, vertex_usage
from evaluation.SAS.SMFR.run_baseline import messages_for, selected_vote
from evaluation.expert_MAS_for_smfr_dataset import pipeline as expert
from evaluation.expert_MAS_for_smfr_dataset.client import TypedExpertClient
from evaluation.simple_MAS_for_smfr_dataset import pipeline as simple
from evaluation.run_mas import instrument


META = dict(investors=['Alice'], question_type='sell', target_percentage=1., aggregation='latest', price_type='Close')
TRANSACTIONS = [dict(company='Apple', date='January 1, 2026', action='buy', quantity=10, price=100.)]
TARGET = dict(total_cost=1000., realized_pnl=0., target_profit=10., profit_needed=10., required_price=101.)


def mock_provider(values, invalid=False):
    provider = object.__new__(Provider)
    provider.name, provider.model = 'openai', 'openai/gpt-oss-120b'
    provider.input_rate, provider.output_rate = .15, .6
    provider.semaphore = asyncio.Semaphore(8)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        fmt = kwargs.get('response_format', {})
        name = (fmt.get('json_schema') or {}).get('name')
        value = values(name, kwargs)
        text = 'malformed response' if invalid else json.dumps(value)
        raw = dict(id=str(len(calls)), choices=[dict(message=dict(content=text), finish_reason='stop')],
                   usage=dict(prompt_tokens=100, completion_tokens=20,
                              completion_tokens_details=dict(reasoning_tokens=10)))
        return SimpleNamespace(model_dump=lambda **kwargs: raw)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return provider, calls


class AnswerTests(unittest.TestCase):
    def test_no_winner_missing_and_false_ties(self):
        self.assertEqual(answer_key(None), ())
        self.assertIsNone(parsed_answer({}))
        self.assertEqual(parsed_answer({'answer': None}), ())
        self.assertEqual(vote_keys([(), (), ('alice',)]), ())
        self.assertEqual(vote_keys([('bob',), ('alice',)]), ('bob',))
        expected = dict(investor_dates={'Alice': [], 'Bob': []}, answer=['Alice'])
        self.assertTrue(compare_answers(expected, dict(answer='Alice')))
        self.assertFalse(compare_answers(expected, dict(answer='Alice and Bob')))

    def test_historical_vote_is_distinct(self):
        draws = [{'answer': None}, {'answer': None}, {'answer': 'Alice'}]
        self.assertEqual(selected_vote(draws, 'canonical_name_sets'), ())
        self.assertEqual(selected_vote(draws, 'historical_multiset'), ('alice',))

    def test_prompt_substitution_preserves_problem_braces(self):
        spec = {'messages': [{'role': 'user', 'content': 'Question: {{problem}}'}]}
        self.assertEqual(messages_for(spec, 'x = {"a": 1}')[0]['content'], 'Question: x = {"a": 1}')

    def test_vertex_thinking_is_counted(self):
        usage = vertex_usage(dict(usage_metadata=dict(prompt_token_count=100, candidates_token_count=20,
                                                     thoughts_token_count=30, total_token_count=150)))
        self.assertEqual(usage['output_tokens'], 50)
        self.assertEqual(usage['thinking_tokens'], 30)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_baseline_complete_run_executes_saved_code_and_resumes(self):
        from evaluation.SAS.SMFR import run_baseline
        provider, calls = mock_provider(lambda name, kwargs: dict(thinking='fixture', answer='Alice',
                                code='def solve():\n    return {"answer": ["Alice"]}'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_path = root / 'input.jsonl'
            samples = [dict(problem='Fixture ' + str(i), answer=dict(investor_dates={'Alice': []}, answer=['Alice']),
                            generation_params=dict(num_investors=2)) for i in range(2)]
            input_path.write_text(''.join(json.dumps(r) + '\n' for r in samples))
            args = argparse.Namespace(protocol='controlled_code_cot', provider='openai', model=provider.model,
                input=input_path, investors=None, limit=None, k=3, max_output_tokens=None,
                output=root / 'outputs', run_id=0, seed=0, execute_code=True, timeout=2,
                workers=2, input_rate=None, output_rate=None, dry_run=False)
            with patch.object(run_baseline, 'Provider', return_value=provider):
                await run_baseline.run(args)
                await run_baseline.run(args)
            result = json.loads((args.output / 'complete.json').read_text())
            self.assertEqual(result['completed_cases'], 2)
            self.assertEqual(result['text_correct'], 2)
            self.assertEqual(result['execution_correct'], 2)
            self.assertEqual(len(calls), 6)

    async def test_typed_expert_pipeline_and_cached_resume(self):
        def values(name, kwargs):
            return {'ProblemMetaResponse': META,
                    'TransactionsResponse': dict(result=TRANSACTIONS, reasoning='fixture'),
                    'CalculationResponse': dict(result=TARGET, reasoning='fixture'),
                    'DatesResponse': dict(result=['January 2, 2026'], reasoning='fixture')}[name]
        provider, calls = mock_provider(values)
        instrument(expert)
        with tempfile.TemporaryDirectory() as folder:
            for _ in range(2):
                client = TypedExpertClient(provider, folder, max_tokens=32768, reasoning_effort='medium')
                answer, stats, trace = await expert.run_pipeline('Fixture problem', client)
                self.assertEqual(answer['answer'], ['Alice'])
                self.assertEqual(stats.llm_calls, 4)
                self.assertEqual(stats.output_tokens, 80)
                self.assertTrue(stats.to_dict()['cost_usd']['pricing_found'] is False or stats.to_dict()['cost_usd']['total'] >= 0)
                self.assertEqual(len(calls), 4)
            usage = [json.loads(p.read_text()) for p in Path(folder).glob('calls/*/usage.json')]
            stages = {u['metadata']['stage'] for u in usage}
            self.assertEqual(stages, {'retrieval', 'calculation', 'dates', 'meta'})
            calc = next(u for u in usage if u['metadata']['stage'] == 'calculation')
            dates = next(u for u in usage if u['metadata']['stage'] == 'dates')
            self.assertEqual(calc['metadata']['branch'], dates['metadata']['branch'])
        for call in calls:
            self.assertEqual(call['reasoning_effort'], 'medium')
            self.assertNotIn('temperature', call)
            self.assertTrue(call['response_format']['json_schema']['strict'])

    async def test_simple_has_three_calls_and_no_financial_scaffolding(self):
        sequence = []
        def values(name, kwargs):
            sequence.append(name)
            if name == 'ProblemMetaResponse':
                return META
            if len(sequence) == 2:
                return {'Alice': TRANSACTIONS}
            return dict(investor_dates={'Alice': ['January 2, 2026']}, answer='Alice')
        provider, calls = mock_provider(values)
        with tempfile.TemporaryDirectory() as folder:
            client = PipelineClient(provider, folder, temperature=0.)
            answer, stats, trace = await simple.run_pipeline('Fixture problem', client)
            self.assertEqual(answer['answer'], 'Alice')
            self.assertEqual(stats.llm_calls, 3)
            self.assertEqual(stats.to_dict()['output_tokens'], 60)
            self.assertEqual(sequence, ['ProblemMetaResponse', None, None])
            self.assertIn('Fixture problem', calls[-1]['messages'][0]['content'])

    async def test_malformed_return_is_cached_without_resampling(self):
        provider, calls = mock_provider(lambda name, kwargs: {}, invalid=True)
        from evaluation.expert_MAS_for_smfr_dataset.agents import ProblemMetaResponse
        with tempfile.TemporaryDirectory() as folder:
            for _ in range(2):
                client = PipelineClient(provider, folder)
                result, usage = await client.generate('Fixture', ProblemMetaResponse)
                self.assertIsNone(result)
                self.assertEqual(usage.output_tokens, 20)
            self.assertEqual(len(calls), 1)

    async def test_transient_retries_are_bounded_and_resume_saved_success(self):
        import httpx
        provider, calls = mock_provider(lambda name, kwargs: META)
        original = provider.client.chat.completions.create
        attempts = []
        def flaky(**kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise httpx.ReadTimeout('fixture')
            return original(**kwargs)
        provider.client.chat.completions.create = flaky
        with tempfile.TemporaryDirectory() as folder, patch('evaluation.common.providers.time.sleep'):
            await provider.complete([dict(role='user', content='fixture')], folder)
            await provider.complete([dict(role='user', content='fixture')], folder)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(len(calls), 1)
            with self.assertRaises(RuntimeError):
                await provider.complete([dict(role='user', content='changed')], folder)


class EvaluationTests(unittest.TestCase):
    def test_saved_answer_and_execution_votes_use_full_reference_denominator(self):
        import contextlib
        import io
        from scripts import evaluate
        from evaluation.common.data import digest
        references = [dict(problem='Case ' + str(i),
                           answer=dict(investor_dates={'Alice': [], 'Bob': []},
                                       answer=['Alice'] if i < 2 else None)) for i in range(3)]
        predictions = [dict(problem_sha256=digest('Case 0'), output={'answer': ['Bob']},
                            execution_vote=['Alice']),
                       dict(problem_sha256=digest('Case 2'), output={'answer': []},
                            execution_vote=None)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ref, out = root / 'reference.jsonl', root / 'outputs.jsonl'
            ref.write_text(''.join(json.dumps(r) + '\n' for r in references))
            out.write_text(''.join(json.dumps(r) + '\n' for r in predictions))
            for metric in ['answer', 'execution']:
                capture = io.StringIO()
                with patch('sys.argv', ['evaluate', '--outputs', str(out), '--reference', str(ref),
                                        '--metric', metric]), contextlib.redirect_stdout(capture):
                    evaluate.main()
                report = json.loads(capture.getvalue())
                self.assertEqual(report['reference_cases'], 3)
                self.assertEqual(report['missing_outputs'], 1)
                self.assertEqual(report['correct'], 1)
            predictions[0].pop('execution_vote')
            out.write_text(json.dumps(predictions[0]) + '\n')
            with patch('sys.argv', ['evaluate', '--outputs', str(out), '--reference', str(ref),
                                    '--metric', 'execution']), self.assertRaisesRegex(ValueError, 'Execution vote missing'):
                evaluate.main()


class ExecutionTests(unittest.TestCase):
    def test_self_contained_program_and_timeout(self):
        result = execute('def solve():\n    return {"answer": ["Alice"]}')
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], {'answer': ['Alice']})
        self.assertEqual(execute('def solve():\n    while True: pass', timeout=.2)['error'], 'timeout')
        self.assertEqual(execute('import subprocess\ndef solve(): return 1')['error'], 'unsupported_import')


if __name__ == '__main__':
    unittest.main()
