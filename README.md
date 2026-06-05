# Smarter Saboteurs, Better Fixers: Scaling & Security in Linear Multi-Agent Workflows

Code and experimental data for our paper at the **ICML 2026 AIWILD Workshop**.

We study how model scale affects the resilience of a strictly linear,
MetaGPT-style software-development pipeline (Product Manager → Architect →
Project Manager → Engineer) when a single Engineer agent is compromised and
instructed to inject subtle, test-evading bugs. We evaluate Pass@1 on HumanEval
across two open-weight model families (Qwen 3.5 and Gemma 3, ~0.3B–27B) and find
a **compliance–correction symmetry**: without downstream correction, the
malicious Pass@1 drop grows with scale (up to 53.7pp at Qwen-27B), but appending
a lightweight terminal QA → Fixer stage — without breaking strict linearity —
collapses that gap to ≤0.6pp and restores statistical parity with control
performance.

## Repository structure

```
ARCC/
├── experiment_runner_ARCC.py                  # Simplified prompting, No-Fix
├── experiment_runner_qa_fix_ARCC.py           # Simplified prompting, QA+Fixer
├── experiment_runner_native_mgpt_ARCC.py      # Native prompting,    No-Fix
├── experiment_runner_native_mgpt_qa_fix_ARCC.py  # Native prompting, QA+Fixer
├── experiment_config.py                       # Roles, prompts, generation config
├── malicious_engineer_profile_new.txt         # Static "Mischievous Engineer" profile
├── generated_malicious_profiles_collection.txt# AutoTransform exploration (see paper App. D)
├── evaluate_results.py                         # Pass@1 scoring (Gemma runs)
├── evaluate_qwen_results.py                    # Pass@1 scoring (Qwen runs)
├── McNemar.py                                  # Paired significance test (builds pass/fail cache)
├── holm_bonferroni.py                          # Holm–Bonferroni correction over McNemar p-values
├── visualize_results.py                        # Scaling / comparison plots
├── requirements.txt
├── data/
│   └── HumanEval.jsonl                         # OpenAI HumanEval benchmark (164 problems)
└── results/
    ├── *_results_*.jsonl                       # Per-problem extracted code + pass/fail
    ├── *_traces_*.jsonl                        # Full per-role generation logs
    ├── pass_fail_cache.json                    # Cached paired outcomes for McNemar
    ├── *-results-*.txt                         # Pass@1 summary tables
    └── plots/                                  # Generated figures
```

The four runners correspond to the paper's 2×2 design: {Native, Simplified}
prompting × {No-Fix, QA+Fixer} correction. Each runs both the Control and
Malicious Engineer scenarios.

## Setup

Requires Python 3.11+ and a CUDA GPU (experiments were run on a single NVIDIA
H100). Models are run locally via HuggingFace Transformers.

```bash
cd ARCC
pip install metagpt==0.8.1   # install alone first to avoid slow resolver
pip install -r requirements.txt
```

MetaGPT 0.8.1 pins older versions of a few packages (numpy, tiktoken,
typing-extensions, ...) that are overridden in `requirements.txt` by the versions
required for Gemma 3 / Qwen 3.5. The resulting pip conflict warnings are expected
and benign — the newer versions win, and generation does not go through MetaGPT's
own LLM stack.

MetaGPT builds a config object at import time and requires `config/config2.yaml`
to exist. A minimal one is included (`ARCC/config/config2.yaml`) with placeholder
LLM settings; it is never used for API calls because generation runs locally via
HuggingFace. No real API keys are needed. A `.env` file is loaded if present but
is not required.

## Running experiments

Select which models to run by editing the `LOCAL_MODELS` list at the top of a
runner (entries are commented/uncommented), then launch it:

```bash
cd ARCC
python experiment_runner_ARCC.py                 # Simplified, No-Fix
python experiment_runner_qa_fix_ARCC.py          # Simplified, QA+Fixer
python experiment_runner_native_mgpt_ARCC.py     # Native, No-Fix
python experiment_runner_native_mgpt_qa_fix_ARCC.py  # Native, QA+Fixer
```

Each run appends per-problem results and full role-by-role traces to
`results/`. Decoding is greedy (`temperature=0.0`, `max_new_tokens=2048`) so
runs are deterministic.

## Testing the pipeline (no GPU required)

`test_pipeline.py` mocks the HuggingFace model to verify MetaGPT role sequencing,
Control/Malicious scenario routing, and results I/O on any machine:

```bash
cd ARCC
python test_pipeline.py   # prints PASS in a few seconds
```

To run a quick end-to-end check on real hardware with fewer than 164 problems,
set the `ARCC_MAX_TASKS` environment variable:

```bash
ARCC_MAX_TASKS=5 python experiment_runner_ARCC.py
```

## Evaluation and analysis

```bash
cd ARCC
python evaluate_results.py        # Pass@1 for Gemma runs
python evaluate_qwen_results.py   # Pass@1 for Qwen runs
python McNemar.py                 # Paired significance test (use --force to rebuild the cache)
python holm_bonferroni.py         # Multiple-comparison correction
python visualize_results.py       # Plots
```

## Data and attribution

- **HumanEval** (`ARCC/data/HumanEval.jsonl`) is the benchmark released by OpenAI
  under the MIT License (Chen et al., 2021, *Evaluating Large Language Models
  Trained on Code*).
- The linear MetaGPT setup and threat model build on **Huang et al. (2024)**,
  *On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents*,
  whose work we compare against. Their code is not redistributed here; see their
  repository for the AutoTransform baseline referenced in the paper.
- Agent orchestration uses **MetaGPT** (Hong et al., 2024), installed as a
  dependency.

## Citation

```bibtex
@inproceedings{mcallister2026saboteurs,
  title     = {Smarter Saboteurs, Better Fixers: Scaling \& Security in Linear Multi-Agent Workflows},
  author    = {McAllister, Timothy and Abdidizaji, Sina and Garibay, Ivan and Ozmen Garibay, Ozlem},
  booktitle = {ICML 2026 Workshop on Agentic AI for Science and Engineering in the Wild (AIWILD)},
  year      = {2026}
}
```

## License

Released under the MIT License (see [LICENSE](LICENSE)). Third-party components
(HumanEval, MetaGPT) are governed by their own licenses.
