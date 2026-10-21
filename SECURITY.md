# Security Policy

## Supported versions

This repository is the code and data accompanying the paper *"On improving
experimental binding affinity predictions with synthetic data."* It is provided
for research and reproducibility purposes and is **not intended for production
use**. Only the latest state of the `main` branch is maintained.

## Reporting a vulnerability

If you discover a security vulnerability, please **do not open a public
issue**. Instead, report it privately using GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
for this repository (Security tab → "Report a vulnerability").

We will acknowledge your report and work with you to understand and address the
issue.

## Security considerations for users

### Pipeline workspaces (`pipeline.pkl.gz`)

`scripts/pcm_pipeline.py train` saves the fitted pipeline to
`<output-dir>/pipeline.pkl.gz` with `pickle`, and `scripts/pcm_pipeline.py predict`
loads `<workspace>/pipeline.pkl.gz` back with `pickle.load`.

**Deserializing a pickle file executes arbitrary code.** Only pass `--workspace`
directories that you created yourself or obtained from a trusted source. Never
load a workspace from an untrusted or unverified third party.
