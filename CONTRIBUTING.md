# Contributing to Automated-MAS-Illusion

This repository accompanies the SMFR evaluation paper. Contributions should
preserve the canonical dataset, explicit experimental protocols, and the
distinction between historical and newer baseline configurations.

## Report an issue

Use [GitHub Issues](https://github.com/SalesforceAIResearch/Automated-MAS-Illusion/issues)
for reproducible bugs or proposed improvements. Include the command, Python
and dependency versions, relevant manifest, and a minimal example. Remove API
keys, tokens, private endpoints, and personal information from logs. Report
security vulnerabilities through [SECURITY.md](SECURITY.md), not public issues.

## Submit a change

1. Create a branch for the change and keep the scope focused.
2. Do not add automated MAS framework source, third-party benchmark datasets,
   live credentials, provider logs, or mutable experiment outputs.
3. Keep canonical test labels/data unchanged unless the change documents a
   verified correction and updates checksums and affected evaluation evidence.
4. Add relevant offline tests for changes to scoring, voting, prompts, API
   checkpointing, or pipeline logic.
5. Run `python -m scripts.verify_release` and
   `python -m unittest discover -s tests -v`.
6. Open a pull request against `main`, explaining the behavior and validation.
   Changes require peer review before merging.

## Contributor License Agreement

Contributors must sign the [Salesforce CLA](https://cla.salesforce.com/sign-cla).
By contributing code, you agree to the project's existing
[CC BY-NC 4.0 license](LICENSE.txt). The dataset uses the same license;
see [dataset/LICENSE](dataset/LICENSE).

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).
