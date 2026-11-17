# Simple-MAS: context-separation anchor

Simple-MAS performs three sequential LLM calls:

1. **MetaAgent** extracts investor names, target percentage, question type,
   aggregation, and price type using the shared meta schema.
2. **RetrieverAgent** (the `RetrievalAgent` implementation) extracts transactions
   and transaction-date prices for all investors.
3. **CalculatorAgent** (the `CalculationAgent` implementation) identifies open
   positions, calculates financial targets, selects dates, and returns winners.

Both retrieval and calculation receive the original problem. The calculation
call additionally receives the parsed retrieval result. No deterministic
financial calculation, date comparison, or final winner aggregation is added.
The pipeline parses/normalizes output JSON but does not execute generated code.

Prompts are in [prompts/](prompts/); [pipeline.py](pipeline.py) preserves the
experiment pipeline. The shared MetaAgent uses Expert-MAS's `meta_agent.txt`;
the copy in this directory is identical for inspection.

```bash
python -m evaluation.run_mas --system simple \
  --provider vertex --model gemini-2.5-pro \
  --limit 2 --output outputs/simple_smoke --dry-run
```

Remove `--dry-run` to make requests. See the root README for providers,
checkpointing, token accounting, and repetition configuration.
