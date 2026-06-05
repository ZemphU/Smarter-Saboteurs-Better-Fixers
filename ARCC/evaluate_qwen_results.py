import json
import os
import re
import multiprocessing
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent

def check_solution_worker(full_code, result_queue):
    """
    Worker function to execute code in a separate process.
    """
    try:
        # Create a new namespace for execution
        exec_globals = {}
        exec(full_code, exec_globals)
        result_queue.put("success")
    except Exception as e:
        result_queue.put(str(e))

def run_with_timeout(full_code, timeout=3.0):
    """
    Runs the code in a separate process with a timeout.
    """
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=check_solution_worker, args=(full_code, queue))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Timeout"
    
    if not queue.empty():
        result = queue.get()
        if result == "success":
            return True, None
        else:
            return False, result
    return False, "Process finished without result"

def evaluate_results(results_file, output_txt=None):
    """
    Evaluates the results from the experiment.
    Calculates Pass@1 rate for each model and scenario.
    Optionally writes the report to a .txt file.
    """
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return

    # Load HumanEval tasks for reference (test cases)
    tasks = {}
    benchmark_path = os.path.join(project_root, 'data', 'HumanEval.jsonl')
    try:
        with open(benchmark_path, 'r') as f:
            for line in f:
                task = json.loads(line)
                tasks[task['task_id']] = task
    except FileNotFoundError:
        print(f"Benchmark file not found at {benchmark_path}")
        return

    # Load Results
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(tasks)} tasks from benchmark.")
    print(f"Evaluating {len(results)} results from {os.path.basename(results_file)}...")
    metrics = {}

    for result in results:
        task_id = result['task_id']
        model = result['model']
        scenario = result['scenario']

        if model not in metrics:
            metrics[model] = {}
        if scenario not in metrics[model]:
            metrics[model][scenario] = {'passed': 0, 'total': 0}
        output = result['output']
        
        if result['status'] != 'success':
            metrics[model][scenario]['total'] += 1
            continue

        # Extract Code
        # Simple extraction: look for python code blocks
        code_match = re.search(r'```python\n(.*?)```', output, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            # Fallback: assume the whole output is code if no blocks, or just use it as is
            code = output

        # Get Test Case
        task = tasks.get(task_id)
        if not task:
            print(f"Task {task_id} not found in benchmark.")
            continue
            
        test_code = task['test']
        entry_point = task['entry_point']
        
        # Combine Code and Test
        full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"
        
        passed, error = run_with_timeout(full_code, timeout=3.0)
        
        if passed:
            metrics[model][scenario]['passed'] += 1
        else:
            pass
        
        metrics[model][scenario]['total'] += 1

    # Build Report
    report_lines = []
    report_lines.append("--- Evaluation Report ---")
    for model, scenarios in metrics.items():
        report_lines.append(f"\nModel: {model}")
        for scenario, data in scenarios.items():
            passed = data['passed']
            total = data['total']
            rate = (passed / total * 100) if total > 0 else 0
            report_lines.append(f"  Scenario: {scenario} | Passed: {passed}/{total} | Rate: {rate:.2f}%")

    report_text = "\n".join(report_lines)

    # Print to console
    print(f"\n{report_text}")

    # Write to .txt file if specified
    if output_txt:
        with open(output_txt, 'w') as f:
            f.write(report_text + "\n")
        print(f"\nResults saved to: {output_txt}")

if __name__ == "__main__":
    results_dir = os.path.join(project_root, 'results')

    # All 4 Qwen results files mapped to their output .txt names
    qwen_files = {
        'qwen_results_arcc.jsonl':                    'qwen-results-nofix.txt',
        'qwen_results_native_mgpt_arcc.jsonl':        'qwen-native-results-nofix.txt',
        'qwen_results_native_mgpt_qa_fix_arcc.jsonl': 'qwen-native-results-qafix.txt',
        'qwen_results_qa_fix_arcc.jsonl':              'qwen-results-qafix.txt',
    }

    for jsonl_file, txt_file in qwen_files.items():
        results_path = os.path.join(results_dir, jsonl_file)
        output_path = os.path.join(results_dir, txt_file)
        print(f"\n{'='*60}")
        print(f"Processing: {jsonl_file}")
        print(f"{'='*60}")
        evaluate_results(results_path, output_txt=output_path)
