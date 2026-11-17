"""Recompute released accuracies/SDs and optional paired McNemar tests offline."""
import argparse
from collections import defaultdict
import csv
import json
import math
import statistics

from evaluation.common.data import ROOT, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--a', help='Method A for an optional paired test, e.g. Simple-MAS')
    parser.add_argument('--b', default='CoT-SC')
    parser.add_argument('--run-a')
    parser.add_argument('--run-b')
    args = parser.parse_args()
    groups = defaultdict(dict)
    for row in read_jsonl(ROOT / 'results/paired_correctness.jsonl.gz'):
        if row['scope'] != 'test' or (args.model and row['model'] != args.model):
            continue
        groups[row['model'], row['method'], row['run']][row['problem_sha256']] = int(row['correct'])
    summaries = defaultdict(list)
    for (model, method, run), vector in groups.items():
        if len(vector) != 588:
            raise ValueError('Incomplete test repetition: ' + str((model, method, run)))
        summaries[model, method].append(100 * sum(vector.values()) / 588)
    output = []
    for (model, method), values in sorted(summaries.items()):
        output.append(dict(model=model, method=method, runs=len(values), cases=588,
                           accuracy_pct=statistics.mean(values),
                           sample_sd_pp=statistics.stdev(values) if len(values) > 1 else None))
    report = dict(summary=output)
    if args.a:
        if not args.model:
            parser.error('--model is required for a paired test')
        def choose(method, requested):
            candidates = [(run, vector) for (model, name, run), vector in groups.items()
                          if model == args.model and name == method and (requested is None or run == requested)]
            if not candidates:
                parser.error('No complete run found for ' + method)
            return max(candidates, key=lambda item: sum(item[1].values()))
        run_a, a = choose(args.a, args.run_a)
        run_b, b = choose(args.b, args.run_b)
        if a.keys() != b.keys():
            raise ValueError('Paired tests require identical problem sets')
        wins = sum(a[k] and not b[k] for k in a)
        losses = sum(b[k] and not a[k] for k in a)
        discordant = wins + losses
        exact = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2 ** discordant) if discordant else 1.
        cc = math.erfc(max(0, abs(wins - losses) - 1) / math.sqrt(2 * discordant)) if discordant else 1.
        report['paired_test'] = dict(model=args.model, a=args.a, b=args.b, run_a=run_a, run_b=run_b,
                                     cases=len(a), a_only_correct=wins, b_only_correct=losses,
                                     exact_two_sided_mcnemar_p=exact, continuity_corrected_p=cc,
                                     selection='Explicit runs where provided; otherwise best accuracy per method, matching the historical paper convention.',
                                     multiple_testing='Unadjusted; do not pool repetitions as independent problems.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
