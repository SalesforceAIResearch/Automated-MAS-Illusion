# SMFR data card

SMFR contains synthetic investor-portfolio questions over historical stock-price
tables and transaction events. A model must retrieve transactions and prices,
track open positions, calculate financial targets, select valid dates, and
compare investors. The supplied answer includes `investor_dates`, `comparison`,
and the final winning investor name(s), or no winner.

## Canonical release

| Investors N | Test cases |
|---:|---:|
| 2 | 96 |
| 3 | 96 |
| 4 | 104 |
| 5 | 152 |
| 6 | 140 |
| Total | 588 |

The five `balanced_dataset_single_N_fixed_v2.jsonl.gz` files are exact gzip
representations of the current experiment references, including metadata and
reference reasoning. The 16 validation cases (eight from N=2 and eight from
N=3) are separate in `balanced_dataset_single_validate_fixed.jsonl.gz`.
[MANIFEST.json](MANIFEST.json) records compressed and uncompressed SHA-256
checksums, split roles, and counts. Problems must not overlap between splits.

The `fixed_v2` annotations supersede earlier `fixed`/original labels. Historical
files sometimes include validation records, so score only the canonical test
problem hashes. Benchmark loading and verification do not fetch online data.

## Generate new problems

```bash
pip install -e '.[generation]'
cd dataset/src
python generate_balanced_dataset.py 2
```

The generator uses Yahoo Finance observations through `yfinance`. Regeneration
is not byte-for-byte reconstruction of the canonical release: date windows,
external observations, and revised annotations can differ. New generated files
are a new dataset; retain their configuration and do not overwrite the pinned
evaluation files. The generator also supplies reference reasoning and labels;
keep those fields out of model prompts.

Investor count varies jointly with the haystack/context, transaction structure,
and distractors. This benchmark is synthetic financial reasoning, not a trading
recommendation dataset or a measure of live financial decision quality.

## License and provenance

The existing SMFR metadata declares CC BY 4.0 for the dataset. Attribute SMFR
and its accompanying paper when redistributing it. The published release
preserves the source annotations and recorded historical price values; no
third-party benchmark datasets are bundled. Source-data rights and the code's
CC BY-NC 4.0 license are separate.
