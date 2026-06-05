"""
Holm-Bonferroni correction for McNemar's test results.

Applies the step-down procedure to the raw p-values from McNemar.py
and reports which comparisons survive correction at alpha = 0.05.

Groups: 10 No-Fix tests + 10 QA+Fixer tests per prompting style,
corrected separately (each group has m=10).
"""

# Raw p-values extracted from mcnemar result.txt
# Format: (label, raw_p, b, c)

simplified_nofix = [
    ("Gemma 0.27B", 1.0000,     0,  0),
    ("Gemma 1B",    5.0000e-01, 2,  0),
    ("Gemma 4B",    9.7656e-04, 14, 1),
    ("Gemma 12B",   2.9523e-10, 48, 3),
    ("Gemma 27B",   6.1079e-09, 42, 3),
    ("Qwen 0.8B",   1.0000,     0,  0),
    ("Qwen 2B",     1.2500e-01, 4,  0),
    ("Qwen 4B",     1.2500e-01, 4,  0),
    ("Qwen 9B",     5.5112e-09, 34, 0),
    ("Qwen 27B",    0.0000,     88, 0),  # underflow -> 0
]

simplified_qafixer = [
    ("Gemma 0.27B", 1.0000,     0,  0),
    ("Gemma 1B",    5.0000e-01, 2,  0),
    ("Gemma 4B",    3.1250e-02, 6,  0),
    ("Gemma 12B",   5.7373e-02, 11, 3),
    ("Gemma 27B",   2.8906e-01, 6,  2),
    ("Qwen 0.8B",   1.0000,     0,  0),
    ("Qwen 2B",     1.0000,     0,  0),
    ("Qwen 4B",     1.0000,     0,  0),
    ("Qwen 9B",     1.0000,     1,  0),
    ("Qwen 27B",    1.0000,     1,  0),
]

native_nofix = [
    ("Gemma 0.27B", 1.0000,     0,  0),
    ("Gemma 1B",    1.0000,     1,  0),
    ("Gemma 4B",    2.0921e-02, 16, 32),
    ("Gemma 12B",   5.7330e-07, 33, 3),
    ("Gemma 27B",   1.8646e-06, 33, 4),
    ("Qwen 0.8B",   1.0000,     3,  3),
    ("Qwen 2B",     1.0692e-03, 22, 5),
    ("Qwen 4B",     9.0234e-03, 24, 9),
    ("Qwen 9B",     2.0887e-08, 37, 2),
    ("Qwen 27B",    1.2212e-15, 64, 0),
]

native_qafixer = [
    ("Gemma 0.27B", 1.0000,     0,  0),
    ("Gemma 1B",    1.0000,     1,  0),
    ("Gemma 4B",    6.2979e-02, 12, 23),
    ("Gemma 12B",   7.5391e-01, 4,  6),
    ("Gemma 27B",   5.4883e-01, 7,  4),
    ("Qwen 0.8B",   1.0000,     7,  6),
    ("Qwen 2B",     1.0000,     9,  8),
    ("Qwen 4B",     1.1666e-01, 9,  17),
    ("Qwen 9B",     1.0000,     1,  1),
    ("Qwen 27B",    5.0000e-01, 0,  2),
]


def holm_bonferroni(tests, alpha=0.05):
    """
    Apply Holm-Bonferroni step-down correction.
    
    Args:
        tests: list of (label, raw_p, b, c) tuples
        alpha: family-wise error rate
    
    Returns:
        list of (label, raw_p, threshold, significant, rank) tuples, 
        in original order
    """
    m = len(tests)
    
    # Sort by p-value (ascending)
    indexed = [(i, label, p, b, c) for i, (label, p, b, c) in enumerate(tests)]
    sorted_tests = sorted(indexed, key=lambda x: x[2])
    
    # Step-down: walk through sorted p-values
    results = {}
    all_rejected = True
    for rank, (orig_idx, label, p, b, c) in enumerate(sorted_tests, start=1):
        threshold = alpha / (m - rank + 1)
        
        if all_rejected and p <= threshold:
            significant = True
        else:
            significant = False
            all_rejected = False  # once one fails, all subsequent fail
        
        results[orig_idx] = (label, p, b, c, threshold, significant, rank)
    
    # Return in original order
    return [results[i] for i in range(m)]


def print_group(title, tests, alpha=0.05):
    """Print Holm-Bonferroni results for a group of tests."""
    results = holm_bonferroni(tests, alpha)
    
    print(f"\n{'='*80}")
    print(f"  {title}  (m={len(tests)}, alpha={alpha})")
    print(f"{'='*80}")
    print(f"  {'Model':<14} {'b/c':>7} {'Raw p':>12} {'Rank':>5} {'Threshold':>10} {'Survives?':>10}")
    print(f"  {'-'*14} {'-'*7} {'-'*12} {'-'*5} {'-'*10} {'-'*10}")
    
    for label, p, b, c, threshold, sig, rank in results:
        p_str = f"{p:.4e}" if p > 0 else "< 1e-16"
        sig_str = "YES ***" if sig else "no"
        print(f"  {label:<14} {b:>3}/{c:<3} {p_str:>12} {rank:>5} {threshold:>10.5f} {sig_str:>10}")
    
    sig_count = sum(1 for *_, sig, _ in results if sig)
    print(f"\n  Summary: {sig_count}/{len(tests)} significant after Holm-Bonferroni correction")
    
    # Show the step-down walkthrough
    print(f"\n  Step-down walkthrough (sorted by p-value):")
    sorted_results = sorted(
        [(label, p, b, c, threshold, sig, rank) for label, p, b, c, threshold, sig, rank in results],
        key=lambda x: x[6]  # sort by rank
    )
    for label, p, b, c, threshold, sig, rank in sorted_results:
        p_str = f"{p:.4e}" if p > 0 else "< 1e-16"
        comparison = "<=" if sig else "> "
        result = "REJECT" if sig else "FAIL -> STOP"
        print(f"    Rank {rank:>2}: p={p_str:>12} {comparison} {threshold:.5f}  ({label}) -> {result}")
        if not sig:
            remaining = len(tests) - rank
            if remaining > 0:
                print(f"    (Ranks {rank+1}-{len(tests)}: automatically non-significant)")
            break


if __name__ == "__main__":
    print("HOLM-BONFERRONI CORRECTION FOR MCNEMAR'S TEST RESULTS")
    print("=" * 80)
    
    groups = [
        ("Simplified Prompting — No-Fix",    simplified_nofix),
        ("Simplified Prompting — QA+Fixer",   simplified_qafixer),
        ("Native Prompting — No-Fix",         native_nofix),
        ("Native Prompting — QA+Fixer",       native_qafixer),
    ]
    
    for title, tests in groups:
        print_group(title, tests)
    
    # Final summary matching the paper's tables
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY: Should these cells be shaded in the paper?")
    print("=" * 80)
    
    for title, tests in groups:
        results = holm_bonferroni(tests)
        print(f"\n  {title}:")
        for label, p, b, c, threshold, sig, rank in results:
            if p < 1.0:  # skip trivial p=1 cases
                status = "SHADED" if sig else "not shaded"
                p_str = f"{p:.4e}" if p > 0 else "< 1e-16"
                print(f"    {label:<14} p={p_str:<14} -> {status}")
