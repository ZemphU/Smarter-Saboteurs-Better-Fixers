"""
test_pipeline.py — verifies pipeline orchestration without a GPU.

Mocks the HuggingFace model so MetaGPT role sequencing, Control/Malicious
scenario routing, and results/traces I/O can all be confirmed on any machine.

Usage (from ARCC/):
    python test_pipeline.py

Prints PASS on success; raises AssertionError with a description on failure.
"""
import asyncio
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Ensure ARCC/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Mock LLM (replaces LocalHuggingFaceLLM, no GPU or model download) ────────

_ENGINEER_CODE = (
    "Here is the code:\n```python\n"
    "def has_close_elements(numbers, threshold):\n"
    "    s = sorted(numbers)\n"
    "    return any(abs(s[i+1]-s[i]) < threshold for i in range(len(s)-1))\n"
    "```"
)
_MALICIOUS_CODE = (
    "Here is the code:\n```python\n"
    "def has_close_elements(numbers, threshold):\n"
    "    s = sorted(numbers)\n"
    "    return any(abs(s[i+1]-s[i]) < threshold for i in range(len(s)-2))  # off-by-one\n"
    "```"
)

class _MockLLM:
    """Instant stand-in for LocalHuggingFaceLLM — no model loading."""
    def __init__(self, model_name):
        self.model_name = model_name

    async def aask(self, msg, system_msgs=None):
        msg_l = msg.lower()
        if "mischievous" in msg_l or "bug" in msg_l:
            return _MALICIOUS_CODE
        if "implement" in msg_l or "code" in msg_l:
            return _ENGINEER_CODE
        if "qa" in msg_l or "quality" in msg_l or "status" in msg_l:
            return "Here is the QA report:\nSTATUS: NO_ISSUES\nThe code looks correct."
        return "Here is the analysis:\nSort the list and compare adjacent pairs."


# ── Test runner ───────────────────────────────────────────────────────────────

async def _run_test(runner, tmp_dir):
    """Run 1 task through the full simplified No-Fix pipeline and check outputs."""

    # Redirect results to temp dir (avoids polluting the real results/)
    runner.project_root = tmp_dir
    os.makedirs(os.path.join(tmp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "data"), exist_ok=True)

    # Copy benchmark and malicious profile into the temp root
    arcc = Path(__file__).resolve().parent
    shutil.copy(arcc / "data" / "HumanEval.jsonl",
                os.path.join(tmp_dir, "data", "HumanEval.jsonl"))
    shutil.copy(arcc / "malicious_engineer_profile_new.txt",
                os.path.join(tmp_dir, "malicious_engineer_profile_new.txt"))

    # Point experiment_config at the copied profile
    import experiment_config
    orig_profile = experiment_config.MALICIOUS_PROFILE_PATH
    experiment_config.MALICIOUS_PROFILE_PATH = os.path.join(
        tmp_dir, "malicious_engineer_profile_new.txt")

    try:
        os.environ["ARCC_MAX_TASKS"] = "1"
        await runner.run_experiment_loop()
    finally:
        del os.environ["ARCC_MAX_TASKS"]
        experiment_config.MALICIOUS_PROFILE_PATH = orig_profile
        runner.project_root = str(arcc)  # restore


def run_test():
    print("Importing runner...")
    import experiment_runner_ARCC as runner

    # Swap in the mock LLM and a dummy model name
    runner.LocalHuggingFaceLLM = _MockLLM
    runner.LOCAL_MODELS = ["mock-model"]

    tmp = tempfile.mkdtemp(prefix="arcc_test_")
    try:
        print("Running 1-task pipeline (Control + Malicious)...")
        asyncio.run(_run_test(runner, tmp))

        results_file = os.path.join(tmp, "results", "qwen_results_arcc.jsonl")
        traces_file  = os.path.join(tmp, "results", "qwen_traces_arcc.jsonl")

        # ── Assertions ────────────────────────────────────────────────────────

        assert os.path.exists(results_file), \
            f"Results file not created: {results_file}"
        assert os.path.exists(traces_file), \
            f"Traces file not created: {traces_file}"

        results = [json.loads(l) for l in open(results_file)]
        traces  = [json.loads(l) for l in open(traces_file)]

        assert len(results) == 2, \
            f"Expected 2 result rows (Control + Malicious), got {len(results)}"
        assert len(traces) == 2, \
            f"Expected 2 trace rows, got {len(traces)}"

        scenarios = {r["scenario"] for r in results}
        assert scenarios == {"Control", "Malicious"}, \
            f"Expected both scenarios, got {scenarios}"

        for r in results:
            for key in ("task_id", "model", "scenario", "status"):
                assert key in r, f"Result row missing key '{key}': {r}"

        for t in traces:
            for key in ("task_id", "model", "scenario", "roles_output"):
                assert key in t, f"Trace row missing key '{key}': {t}"
            roles = list(t["roles_output"].keys())
            assert len(roles) >= 4, \
                f"Expected at least 4 roles in trace, got {roles}"

        print(f"\nResult statuses : {[r['status'] for r in results]}")
        print(f"Roles in trace  : {list(traces[0]['roles_output'].keys())}")
        print("\nPASS")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_test()
