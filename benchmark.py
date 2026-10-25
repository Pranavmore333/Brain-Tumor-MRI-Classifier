"""
Academic Benchmarking Runner.
Generates evaluation tables comparing models against Clinical Acceptance Thresholds.
"""

from src.evaluation.validator import BenchmarkValidator

if __name__ == "__main__":
    print("[*] Compiling 5-Fold Stratified Cross-Validation Benchmark Suite...")
    report = BenchmarkValidator.generate_capstone_benchmark_report()
    md_table = BenchmarkValidator.format_markdown_table(report)

    print("\n" + "=" * 78)
    print("      ACADEMIC BENCHMARK RESULTS (BraTS Multi-Center Cohort)")
    print("=" * 78)
    print(md_table)
    print("=" * 78 + "\n")
