# Standalone SMFR baselines

Run from the repository root with `python -m evaluation.SAS.SMFR.run_baseline`.
All prompts are plain JSON files in [prompts/](prompts/), with the literal
`{{problem}}` sentinel replaced by the problem text only. Labels and reference
reasoning never enter the prompt. No automated MAS package is imported.

```bash
# Text-only Gemini CoT-SC, one repetition, k=3
python -m evaluation.SAS.SMFR.run_baseline \
  --protocol gemini_cot --provider vertex --model gemini-2.5-pro \
  --k 3 --output outputs/gemini_cot_sc_0

# Code-generating Gemini k=3, with separate execution voting
python -m evaluation.SAS.SMFR.run_baseline \
  --protocol gemini_code_cot --provider vertex --model gemini-2.5-pro \
  --k 3 --execute-code --output outputs/gemini_code_cot_sc_0

# Historical GPT-OSS code-generating prompt, k=5; direct answer metric
python -m evaluation.SAS.SMFR.run_baseline \
  --protocol gptoss_historical --provider together --model openai/gpt-oss-120b \
  --k 5 --output outputs/gptoss_historical_0
```

Set `--k 1` for a single draw. `--limit 2 --dry-run` writes a manifest and one
rendered prompt without credentials or paid requests. `--investors 2 3` selects
depths. An explicit `--input` accepts JSONL or gzip JSONL. Resume with the same
command/output directory; a changed prompt or configuration requires a new
directory. Each API response is checkpointed before parsing or execution.

`--execute-code` is opt-in. The worker supports a standard-library import
allowlist and self-contained `solve()` programs. It is a new standalone
executor with a different policy from some historical ones, not an OS security
sandbox. Use an isolated environment without credentials for generated code.

The `gemini_*` presets preserve the final Gemini prompts. The `gpt5_historical`
preset archives the earlier baseline prompt and k; original serving defaults
were not fully recorded. Controlled/newer presets are labeled separately from
historical baselines. See [the protocol guide](../../../docs/protocols.md).
