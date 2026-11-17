# The Illusion of Automatically Orchestrated Multi-Agent Advantage

This repository releases our **Synthetic Multi-Hop Financial Reasoning (SMFR)**
benchmark, standalone CoT/CoT-SC and Code-CoT/Code-CoT-SC baselines, and
**Simple-MAS** and **Expert-MAS** reference systems.

It contains our dataset generation, prompts, pipelines, evaluation, and result
summaries. Automated MAS framework implementations, trained controllers,
generated framework architectures, and third-party benchmark datasets are not
bundled. Obtain those implementations and datasets from their original authors.

## Three-tier evaluation

The paper introduces three reference tiers for diagnosing where an automated
MAS gains or loses performance. They separate the value of additional sampling,
retrieval/reasoning context separation, and engineered task decomposition.

| Tier | Reference system | Role in the evaluation |
|---|---|---|
| **1 · Cost-efficiency floor** | [CoT-SC](evaluation/SAS/SMFR/README.md) | Multiple single-agent reasoning draws with answer voting establish the reference for judging whether orchestration delivers an accuracy/cost benefit. |
| **2 · Context-separation anchor** | [Simple-MAS](evaluation/simple_MAS_for_smfr_dataset/README.md) | A three-call pipeline—meta parsing, retrieval, and calculation—separates context processing from reasoning. It tests how much can be achieved through this basic split, without deterministic financial scaffolding. |
| **3 · Topological-ceiling reference** | [Expert-MAS](evaluation/expert_MAS_for_smfr_dataset/README.md) | Investor-specific decomposition, parallel reasoning branches, and deterministic position tracking and winner aggregation provide an engineered reference for what task-aware orchestration can achieve. |

Comparing an automated MAS with Tier 1 assesses its accuracy/cost tradeoff;
Tier 2 tests whether it captures the gains available from context separation;
Tier 3 characterizes the remaining gap to investor-specific decomposition and
structured scaffolding. **Topological ceiling** names this expert-designed
reference tier; it does not establish a formal performance upper bound.

**Code-CoT-SC** complements these tiers by testing whether code generation and
execution account for performance gains. Comparisons should use the same
backbone, matched test problems, and clearly documented prompt and cost
protocols; see [the protocol guide](docs/protocols.md).

## Install and verify

```bash
git clone https://github.com/SalesforceAIResearch/Automated-MAS-Illusion.git
cd Automated-MAS-Illusion
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m scripts.verify_release
python -m unittest discover -s tests -v
```

Python 3.10+ is required; the release is checked on Python 3.12. Install
`pip install -e '.[vertex]'` for Gemini on Vertex, or
`pip install -e '.[generation]'` to generate new synthetic problems.
[requirements-validated.txt](requirements-validated.txt) records the local
dependency versions used for the offline checks.

The release includes ordinary gzip files, **not Git LFS pointers**. Loaders read
compressed JSONL directly. No model requests are made by verification or tests.

## Contents

| Path | Contents |
|---|---|
| [`dataset/data/`](dataset/README.md) | Exact v2 test set and held-out validation split |
| [`dataset/src/`](dataset/src) | Our synthetic task generator and financial-data source adapters |
| [`dataset/evaluate/scoring.py`](dataset/evaluate/scoring.py) | Canonical final-answer evaluator |
| [`evaluation/SAS/SMFR/`](evaluation/SAS/SMFR/README.md) | Standalone baselines and versioned prompt presets |
| [`evaluation/simple_MAS_for_smfr_dataset/`](evaluation/simple_MAS_for_smfr_dataset/README.md) | Context-separation anchor: meta → retrieval → calculation |
| [`evaluation/expert_MAS_for_smfr_dataset/`](evaluation/expert_MAS_for_smfr_dataset/README.md) | Investor-specific decomposition and deterministic orchestration |
| [`evaluation/SAS/prompts/`](evaluation/SAS/prompts) | Archived task-specific baseline prompt text for other evaluated benchmarks |
| [`results/`](results/README.md) | Aggregate results and paired correctness snapshots |
| [`docs/protocols.md`](docs/protocols.md) | Historical/new protocol distinctions and accounting limitations |

## Dataset

Use the released `fixed_v2` files for the paper's canonical **588-problem test
set**. Investor counts 2–6 contain 96, 96, 104, 152, and 140 problems. The
**16 validation problems are separate and disjoint**; do not add them to the
test denominator. [Checksums and counts](dataset/MANIFEST.json) pin both splits.

```python
from evaluation.common.data import dataset
for investors, sample in dataset():
    problem, gold = sample["problem"], sample["answer"]
```

Increasing investor count also changes context size, transaction count, and
other generation parameters; it is not an isolated intervention on one factor.
See the [data card](dataset/README.md) for provenance and generation limits.

## Configure a provider

Copy `.env.example` to `.env`. Public runners support OpenAI-compatible APIs,
Together, OpenRouter, and native Vertex AI. All credentials and outputs stay
local and are ignored by Git.

For Vertex, set `GOOGLE_CLOUD_PROJECT` and authenticate Application Default
Credentials with `gcloud auth application-default login`. No organization
project or internal gateway is hardcoded in this release.

## Run the baselines

Validate configuration without making requests:

```bash
python -m evaluation.SAS.SMFR.run_baseline \
  --protocol gemini_cot --provider vertex --model gemini-2.5-pro \
  --k 3 --limit 2 --output outputs/cot_sc_dry --dry-run
```

Run one Gemini CoT-SC repetition:

```bash
python -m evaluation.SAS.SMFR.run_baseline \
  --protocol gemini_cot --provider vertex --model gemini-2.5-pro \
  --k 3 --workers 8 --output outputs/gemini_cot_sc_run0
```

For Code-CoT-SC, use `--protocol gemini_code_cot --execute-code`. For CoT or
Code-CoT, set `--k 1`. Model-answer voting and executed-code voting are reported
separately. Code execution uses an independent timeout worker with a standard
library allowlist; it is **not an OS security sandbox**. Run model programs in
an isolated environment without credentials. The public worker's import policy
differs from some historical executors; do not silently mix re-executed scores.

Historical GPT-OSS CoT-SC requested code as well as an answer. The
`gptoss_historical` preset preserves that prompt and direct-voting policy. The
`controlled_*` presets are separate experiments; they do not replace the
historical comparison against automated frameworks. Read [protocols.md](docs/protocols.md).

## Run Simple-MAS or Expert-MAS

```bash
python -m evaluation.run_mas \
  --system simple --provider vertex --model gemini-2.5-pro \
  --workers 4 --output outputs/gemini_simple_run0

python -m evaluation.run_mas \
  --system expert --provider together --model openai/gpt-oss-120b \
  --workers 4 --max-calls 16 --output outputs/gptoss_expert_run0
```

Use `--limit 2` for a small run or `--dry-run` for an offline configuration
check. Each invocation runs **one repetition**. Choose a new output directory
and `--run-id` for another repetition. Reusing the same directory resumes
checkpointed responses and verifies that the configuration has not changed.
Create an `outputs/.../STOP` file to stop scheduling new work; active requests
finish. Transient API failures get at most six attempts; malformed successful
responses are retained, never resampled to obtain a valid answer.

Every call stores its request, raw response, usage, timing, and dependency
metadata. Vertex accounting includes thinking tokens. Supply `--input-rate`
and `--output-rate` in USD per million tokens to compute a cost estimate when
the provider does not report cost; otherwise cost is explicitly unavailable.
Pipeline `stats.cost_usd` retains historical rate-card estimates for provenance;
the run summary uses the provider ledger, not that legacy estimate. Neither is
an invoice. Failed requests with unknown billing remain in attempt records.

## Evaluate saved outputs

```bash
python -m scripts.evaluate --outputs outputs/gemini_simple_run0/outputs.jsonl
```

Predictions are joined to canonical references by stripped-problem SHA-256.
Missing outputs count incorrect; false ties are penalized; explicit no-winner
answers are distinguished from missing/malformed answers.

## License and attribution

Our code is distributed under the existing [CC BY-NC 4.0 license](LICENSE.txt). The original SMFR dataset
metadata declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
see [dataset/LICENSE](dataset/LICENSE) and [NOTICE](NOTICE). Financial price
observations were sourced through Yahoo Finance; new generation may obtain
different or revised observations. Use the pinned release files for evaluation.
