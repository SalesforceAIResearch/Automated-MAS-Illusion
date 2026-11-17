# Experimental protocols and limits of reproduction

The release separates the **prompts and task logic** from public-provider
transport. Provider defaults, model versions and gateway behavior can change.
Matching a prompt does not guarantee reproducing a numerical paper result.
Keep each repetition's manifest and raw responses.

## Baselines

| Preset | Scope | Default k | Generation defaults |
|---|---|---:|---|
| `gemini_cot` | Final Gemini text-only CoT/CoT-SC | 3 | temperature 1; 65,536 output-token cap; dynamic thinking; no seed |
| `gemini_code_cot` | Final Gemini code-generating baseline | 3 | Same sampling settings, separate code prompt |
| `gptoss_historical` | Historical GPT-OSS baseline used against automated MAS | 5 | Frozen code-generating prompt/schema and recorded request controls |
| `gpt5_historical` | Historical GPT-5 CoT-SC prompt | 5 | Prompt archived; some original server defaults were not pinned |
| `controlled_cot` | New text-only control | 3 | GPT-5/GPT-OSS: medium reasoning, 32,768 cap; GPT-4o: temperature .8, 16,384 cap |
| `controlled_code_cot` | New code-generating control | 3 | Same controls as `controlled_cot`, with code requested |

Use k=1 for CoT. CoT-SC votes on model answers; Code-CoT-SC can additionally
execute each saved program and vote on its returned answer. These are separate
metrics from the **same paid generations**, not additive model spending.

Historical GPT-OSS CoT-SC generated answer **and code**, even when only the text
answer was reported. The historical direct vote filters missing/empty answers
and votes on name multisets. This differs from the final Gemini baseline,
where explicit no-winner values are valid votes and malformed responses abstain.
The public execution-vote path uses canonical name sets. Some historical code
ensembles used additional structural filters; the archived 35.49% k3 score is
not a promised output of the new public executor/voting implementation.

All voting uses only generated answers, never the ground-truth label. Tied
vote counts choose the earliest surviving draw. Sampling seeds are optional;
provider determinism is not assumed. Controlled GPT-5 uses low verbosity;
controlled presets default their seed to `run_id * 100 + draw_index`, overridable
with `--seed`. Gemini presets omit the seed unless explicitly supplied.

The prompt-only JSON files under `evaluation/SAS/prompts/` preserve task-specific
instructions for the other evaluated benchmarks. Their original datasets and
automated-framework infrastructure are not redistributed. The executable
release entry points focus on SMFR.

## Engineered MAS

**Simple-MAS** uses one structured meta call, one retrieval call for all
investors, and one calculation call that performs reasoning, date selection,
and final aggregation. It does not execute generated programs.

**Expert-MAS** shares the meta schema, extracts each investor's transactions and
prices in parallel, identifies open positions in Python, and runs financial
target calculation followed by date retrieval for each open position. Final
date comparison and winner determination use deterministic Python. There is a
barrier after all transaction retrievals; it is not a completely independent
end-to-end pipeline per investor.

The release preserves the prompt templates and financial pipeline logic.
For GPT-OSS Expert-MAS, it includes the corrected **typed structured-output
adapter**: transaction objects, financial targets, and date arrays are enforced
at the API boundary and serialized back to the original pipeline interface.
This is a distinct protocol from the older failing string-valued GPT-OSS runs.
GPT-OSS uses medium reasoning, a 32,768 output-token cap, and omitted temperature
and seed. Simple-MAS retains its prompt-only meta formatting for GPT-OSS.

Gemini Simple-MAS uses temperature 0, a 65,536-token cap, a meta response schema,
and JSON MIME for retrieval/calculation. Gemini Expert-MAS preserves its
historical native response schemas, temperature 0, and omitted output cap and
thinking override. The Vertex pilot uses the same pipeline and original
schemas; only usage accounting now includes hidden thinking tokens.

The standalone transport retains malformed successful responses. It retries
only transient provider/transport failures, at most six attempts. Exhausted
errors stop new scheduling and preserve raw checkpoints for inspection.
No architecture search or automated follow-up repetition is part of a runner.

## Results and cost accounting

The canonical test set is 588 disjoint v2 cases. Each method/run has its own
correctness vector; do not pool repeated runs as additional unique problems.
Reported standard deviations are sample SDs across complete repetitions.

Recorded completion tokens generally include reasoning for GPT-OSS and GPT-5.
Native Vertex usage explicitly sums candidate and thinking tokens (with a
total-token consistency check). **Historical Gemini Expert-MAS recorded only
candidate tokens.** Its historical ~$218 cost therefore excludes thinking.
The $537 figure in the final plot is a **depth-weighted full-run projection**
from a separate 30-case Vertex pilot, not measured full-run spending. The
sampling-only 95% bootstrap interval is approximately $530–544.

The $9.56 GPT-OSS Code-CoT-SC plotted cost is a projection from a different,
incomplete newer-provider configuration; the plotted 35.49% accuracy comes
from the first three historical-setting generations. It is not a matched
measured accuracy/cost pair.

## Latency

The original rebuttal divided Expert-MAS phase output-token totals by investor
count and took the largest phase; this assumes balanced work and does not
measure elapsed latency. Prefer the saved-call dependency-path analysis where
call-level records exist. It sums output tokens along the longest implemented
dependency path and still ignores prefill, throughput differences, API caps,
and overhead. It is not a measured wall-clock speedup.

The current public runners record stage, investor/open-position branch,
per-call token counts, and elapsed request time, enabling that analysis for
future runs. Historical GPT-4o/GPT-5 Expert-MAS outputs retained phase totals
but not per-call usage; exact critical paths cannot be recovered from those
files alone. Stage totals remain valid measurements of generated work.
