"""Score saved outputs without requiring provider credentials or external frameworks."""
import argparse
import json
from pathlib import Path

from dataset.evaluate.scoring import compare_answers
from evaluation.common.data import dataset, digest, read_jsonl, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--metric', choices=['answer', 'execution'], default='answer',
                        help='Score model answers or the separate Code-CoT-SC execution vote')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    references = {digest(r['problem']): (n, r) for n, r in dataset(path=args.reference)}
    predictions = {}
    ignored = 0
    for row in read_jsonl(args.outputs):
        key = row.get('problem_sha256')
        if key is None and isinstance(row.get('input'), dict):
            key = digest(row['input']['problem'])
        if key not in references:
            ignored += 1
            continue
        if key in predictions:
            raise ValueError('Duplicate prediction for ' + key)
        if args.metric == 'execution':
            if 'execution_vote' not in row:
                raise ValueError('Execution vote missing; use baseline outputs generated with --execute-code')
            vote = row['execution_vote']
            predictions[key] = {'answer': vote} if vote is not None else {}
        else:
            predictions[key] = row.get('output', row.get('parsed', {}))
    correct = {key: key in predictions and compare_answers(r['answer'], predictions[key])
               for key, (_, r) in references.items()}
    report = dict(metric=args.metric, reference_cases=len(references), matched_outputs=len(predictions), ignored_outputs=ignored,
                  missing_outputs=len(references) - len(predictions), correct=sum(correct.values()),
                  accuracy_pct=100 * sum(correct.values()) / len(references),
                  missing_policy='Missing or malformed outputs count incorrect; denominator is the selected reference split.',
                  by_investors=[dict(investors=n, cases=sum(d == n for d, _ in references.values()),
                                     correct=sum(correct[k] for k, (d, _) in references.items() if d == n))
                                for n in sorted({d for d, _ in references.values()}, key=str)])
    if args.report:
        save(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
