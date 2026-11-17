# Released result summaries

These snapshots cover **our standalone baselines and engineered MAS systems**.
No automated framework implementation, generated architecture, or model program
is included. Correctness vectors contain problem hashes and scoring outcomes,
not raw provider responses; the executable runners can produce new raw records.

| File | Contents |
|---|---|
| `smfr_accuracy.csv` | Complete-run mean accuracy and sample SD; pilots explicitly separated |
| `paired_correctness.jsonl.gz` | One correctness bit per method/run/canonical problem for paired analysis |
| `smfr_costs.csv` | Cost values with recorded/projected provenance, not invoices |
| `smfr_by_investors.csv` | Investor-wise accuracy/SD and the original idealized bottleneck proxy |
| `critical_path_tokens.csv` | Corrected token dependency-path estimates; unavailable fields stay empty |
| `stage_output_tokens.csv` | Actual summed output tokens per stage, without division by investor count |
| `MANIFEST.json` | Correctness-snapshot checksum and record count |

Recompute the complete-run summaries without API access:

```bash
python -m scripts.summarize_results
python -m scripts.summarize_results --model Gemini-2.5-Pro --a Simple-MAS --b CoT-SC
```

The paired test defaults to the historical best-versus-best convention unless
explicit `--run-a` and `--run-b` identifiers are supplied. Report the selection
policy and account for multiple testing. The outcomes are paired by canonical
problem hash; repetitions are not pooled as additional unique problems.

Important distinctions:

- Historical CoT-SC and newer controlled prompts are separate methods/protocols.
- GPT-OSS Expert-MAS uses the corrected typed-output runs, not the older
  serialization failures.
- Gemini Simple-MAS and CoT-SC each have two complete repetitions. Its two full
  Expert-MAS runs are historical; the new Vertex Expert pilot has 30 cases.
- Gemini's $537 Expert-MAS cost is a depth-weighted **projection** including
  thinking; it is not the historical full-run bill or a completed third run.
- The GPT-OSS $9.56 Code-CoT-SC cost is from a different paused configuration
  than the historical k3 accuracy. Treat it as an unmatched projection.
- GPT-5 Simple-MAS rescored against canonical v2 labels is **95.35% ±1.32pp**.
  The archived correctness vectors and complete-run averaging support this value.

See [protocols.md](../docs/protocols.md) for token-accounting differences and
limits of historical latency reconstruction. The CSVs distinguish unavailable
measurements from zero. Pilot results are not full-benchmark estimates.
