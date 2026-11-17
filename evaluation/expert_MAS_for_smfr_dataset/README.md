# Expert-MAS: engineered topology reference

Expert-MAS adds investor-specific decomposition and deterministic orchestration
to the shared meta/retrieval/reasoning setup:

1. MetaAgent parses the problem topology and target parameters.
2. One extraction call per investor retrieves transactions and prices in
   parallel. The pipeline waits for all these calls.
3. Python identifies open positions from buy/sell transaction sets.
4. Each open position gets a financial-target calculation, followed by a
   date-selection call. Investor/open-position branches run in parallel.
5. Python sorts dates, compares investors, and determines the winner(s).

Calculation prompts receive one investor's transactions. Transaction retrieval
and date selection still receive the full original problem context. The
scaffolding and dependencies are in [pipeline.py](pipeline.py); prompt text is
in [prompts/](prompts/).

[client.py](client.py) contains the corrected GPT-OSS typed-output adapter used
for the latest complete GPT-OSS runs. Native Gemini retains the original
string-valued stage schemas. These protocols must not be silently pooled with
the older GPT-OSS payload/serialization failures.

```bash
python -m evaluation.run_mas --system expert \
  --provider together --model openai/gpt-oss-120b \
  --limit 2 --output outputs/expert_smoke --dry-run
```

The public client records token usage and branch metadata per call. The
original historical outputs stored only phase totals; dividing those totals
by investor count is an idealized balanced-work estimate, not measured latency.
