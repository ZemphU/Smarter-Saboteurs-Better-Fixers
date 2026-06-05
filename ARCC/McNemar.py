"""
McNemar's test for Control vs. Malicious across all models.
Two-phase approach:
  Phase 1: Evaluate all results and cache pass/fail to a JSON file
  Phase 2: Load cache and compute McNemar's test (instant)

Usage:
    python McNemar.py          # runs both phases (phase 1 skipped if cache exists)
    python McNemar.py --force  # force re-evaluation even if cache exists
"""

import json
import os
import re
import sys
import io
import multiprocessing
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent
CACHE_FILE = project_root / 'results' / 'pass_fail_cache.json'


# ── Phase 1: Evaluate and cache ──

def check_solution_worker(full_code, result_queue):
    """Worker for multiprocessing - suppresses stdout from exec'd code."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec_globals = {}
        exec(full_code, exec_globals)
        result_queue.put("success")
    except Exception as e:
        result_queue.put(str(e))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def run_with_timeout(full_code, timeout=5.0):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=check_solution_worker, args=(full_code, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    if not queue.empty():
        result = queue.get()
        return result == "success"
    return False


def load_benchmark():
    tasks = {}
    path = project_root / 'data' / 'HumanEval.jsonl'
    with open(path, 'r') as f:
        for line in f:
            task = json.loads(line)
            tasks[task['task_id']] = task
    return tasks


def evaluate_single(result, tasks):
    """Returns True/False for whether this result passes."""
    if result['status'] != 'success':
        return False
    task = tasks.get(result['task_id'])
    if not task:
        return False
    output = result['output']
    code_match = re.search(r'```python\n(.*?)```', output, re.DOTALL)
    code = code_match.group(1) if code_match else output
    test_code = task['test']
    entry_point = task['entry_point']
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"
    return run_with_timeout(full_code, timeout=5.0)


ALL_RESULT_FILES = [
    ("Gemma, Simplified, No-Fix",    "experiment_results_arcc.jsonl"),
    ("Gemma, Simplified, QA+Fixer",  "experiment_results_qa_fix_arcc.jsonl"),
    ("Gemma, Native, No-Fix",        "experiment_results_native_mgpt_arcc.jsonl"),
    ("Gemma, Native, QA+Fixer",      "experiment_results_native_mgpt_qa_fix_arcc.jsonl"),
    ("Qwen, Simplified, No-Fix",     "qwen_results_arcc.jsonl"),
    ("Qwen, Simplified, QA+Fixer",   "qwen_results_qa_fix_arcc.jsonl"),
    ("Qwen, Native, No-Fix",         "qwen_results_native_mgpt_arcc.jsonl"),
    ("Qwen, Native, QA+Fixer",       "qwen_results_native_mgpt_qa_fix_arcc.jsonl"),
]


def build_cache():
    """Evaluate all results and save pass/fail to cache file."""
    tasks = load_benchmark()
    print(f"Loaded {len(tasks)} benchmark tasks.")

    # cache structure: { "filename|model|task_id|scenario": true/false }
    cache = {}

    for config_name, filename in ALL_RESULT_FILES:
        filepath = project_root / 'results' / filename
        if not filepath.exists():
            print(f"  Skipping {filename} (not found)")
            continue

        # Load results
        results = []
        with open(filepath, 'r') as f:
            for line in f:
                results.append(json.loads(line))

        total = len(results)
        print(f"\nEvaluating: {config_name} ({filename}) — {total} results")

        for i, result in enumerate(results):
            cache_key = f"{filename}|{result['model']}|{result['task_id']}|{result['scenario']}"
            passed = evaluate_single(result, tasks)
            cache[cache_key] = passed

            if (i + 1) % 50 == 0 or (i + 1) == total:
                sys.stdout.write(f"\r  Progress: {i+1}/{total}")
                sys.stdout.flush()

        print()  # newline after progress

    # Save cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"\nCache saved to {CACHE_FILE} ({len(cache)} entries)")
    return cache


# ── Phase 2: McNemar's test from cache ──

def mcnemar_test(b, c):
    """
    McNemar's test.
    b = control pass & malicious fail (broken by attack)
    c = control fail & malicious pass (accidentally fixed)
    Returns (chi2_stat, p_value, method)
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0, "N/A (no discordant pairs)"

    if n < 25:
        from scipy.stats import binomtest
        result = binomtest(b, n, 0.5)
        return None, result.pvalue, "exact binomial"
    else:
        from scipy.stats import chi2
        stat = (b - c) ** 2 / (b + c)
        p = 1 - chi2.cdf(stat, df=1)
        return stat, p, "chi-squared"


def run_mcnemar(cache):
    """Run McNemar's test using cached pass/fail data."""
    for config_name, filename in ALL_RESULT_FILES:
        # Find all entries for this file in cache
        prefix = f"{filename}|"
        entries = {k: v for k, v in cache.items() if k.startswith(prefix)}
        if not entries:
            continue

        print(f"\n{'='*70}")
        print(f"Config: {config_name}")
        print(f"{'='*70}")

        # Parse entries: extract (model, task_id, scenario) -> pass/fail
        pass_fail = {}
        models_set = set()
        for key, passed in entries.items():
            parts = key.split('|')
            # parts: [filename, model, task_id, scenario]
            model, task_id, scenario = parts[1], parts[2], parts[3]
            pass_fail[(model, task_id, scenario)] = passed
            models_set.add(model)

        models = sorted(models_set)

        for model in models:
            # Get all task_ids for this model
            task_ids = sorted(set(
                k[1] for k in pass_fail.keys()
                if k[0] == model
            ))

            a = b = c = d = 0
            for tid in task_ids:
                ctrl = pass_fail.get((model, tid, 'Control'), False)
                mal  = pass_fail.get((model, tid, 'Malicious'), False)
                if ctrl and mal:
                    a += 1
                elif ctrl and not mal:
                    b += 1
                elif not ctrl and mal:
                    c += 1
                else:
                    d += 1

            stat, p, method = mcnemar_test(b, c)

            sig = ""
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = "n.s."

            short_model = model.split('/')[-1]
            n_total = a + b + c + d
            print(f"\n  {short_model}")
            print(f"    Contingency: a={a}, b={b}, c={c}, d={d}")
            print(f"    Control Pass@1: {(a+b)}/{n_total} = {(a+b)/n_total*100:.1f}%")
            print(f"    Malicious Pass@1: {(a+c)}/{n_total} = {(a+c)/n_total*100:.1f}%")
            print(f"    Discordant: b={b} (broken by attack), c={c} (accidentally fixed)")
            if stat is not None:
                print(f"    McNemar chi2 = {stat:.2f}, p = {p:.4e} ({method}) {sig}")
            else:
                print(f"    p = {p:.4e} ({method}) {sig}")


def main():
    force = '--force' in sys.argv

    if CACHE_FILE.exists() and not force:
        print(f"Loading cached pass/fail data from {CACHE_FILE}")
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(f"  {len(cache)} entries loaded.")
    else:
        print("Building pass/fail cache (this takes a while — one time only)...")
        cache = build_cache()

    print("\n" + "=" * 70)
    print("  McNEMAR'S TEST RESULTS")
    print("=" * 70)
    run_mcnemar(cache)


if __name__ == "__main__":
    main()
