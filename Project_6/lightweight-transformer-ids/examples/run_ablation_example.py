"""Example usage of the ablation study pipeline.

This script demonstrates how to run different ablation studies
and analyze their results programmatically.

Run with: python examples/run_ablation_example.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_ablation import AblationStudy


def run_architecture_ablation():
    """Example: Run architectural ablation study."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Running Architecture Ablation Study")
    print("="*80)

    ablation = AblationStudy(
        config_path="configs/ablation/architecture.yaml",
        output_base="results",
        logs_base="logs"
    )

    # Run only the k sweep experiment
    results = ablation.run_ablation_study(
        data_path="lightweight-transformer-ids/data/raw/ciciot/ciciot_training_50_50.csv",
        experiment_filter="linformer_k_sweep",
        device="cuda"
    )

    print(f"\nCompleted {len(results)} experiments")
    print(f"Results saved to: {ablation.experiment_dir}")

    return ablation, results


def run_design_ablation():
    """Example: Run design choice ablation study."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Running Design Ablation Study")
    print("="*80)

    ablation = AblationStudy(
        config_path="configs/ablation/design.yaml",
        output_base="results",
        logs_base="logs"
    )

    # Run positional encoding ablation
    results = ablation.run_ablation_study(
        data_path="lightweight-transformer-ids/data/raw/ciciot/ciciot_training_50_50.csv",
        experiment_filter="positional_encoding",
        device="cuda"
    )

    print(f"\nCompleted {len(results)} experiments")
    return ablation, results


def analyze_results(ablation, results):
    """Example: Analyze ablation results."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Analyzing Ablation Results")
    print("="*80)

    # Load results from CSV
    import pandas as pd

    results_csv = ablation.experiment_dir / "results.csv"
    df = pd.read_csv(results_csv)

    print("\nResults Summary:")
    print(df.to_string())

    # Find best experiment by F1 score
    if "f1" in df.columns:
        best_idx = df["f1"].idxmax()
        best_result = df.iloc[best_idx]
        print(f"\nBest experiment (by F1): {best_result['experiment']}")
        print(f"F1 Score: {best_result['f1']:.4f}")


def compare_multiple_ablations():
    """Example: Compare results across multiple ablation studies."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Comparing Multiple Ablations")
    print("="*80)

    import pandas as pd
    from pathlib import Path

    results_dir = Path("results/ciciot_binary_classification")

    # Find all ablation studies
    ablation_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    comparison_data = []

    for ablation_dir in ablation_dirs:
        # Find latest timestamp
        timestamp_dirs = sorted([d for d in ablation_dir.iterdir() if d.is_dir()])
        if timestamp_dirs:
            latest = timestamp_dirs[-1]
            csv_file = latest / "results.csv"

            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df["ablation"] = ablation_dir.name
                comparison_data.append(df)

    if comparison_data:
        combined_df = pd.concat(comparison_data, ignore_index=True)
        print("\nCombined Results from All Ablations:")
        print(combined_df.to_string())

        # Find best overall
        if "f1" in combined_df.columns:
            best_idx = combined_df["f1"].idxmax()
            best = combined_df.iloc[best_idx]
            print(f"\nOverall Best: {best['ablation']} / {best['experiment']}")
            print(f"F1 Score: {best['f1']:.4f}")


def demonstrate_benchmarking():
    """Example: Run efficiency benchmarking."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Efficiency Benchmarking")
    print("="*80)

    from pipeline.benchmark import ModelBenchmark
    from src.model import ModelFactory
    from src.config_manager import ModelConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a test model
    model_config = ModelConfig(dim=64, depth=4, heads=4, k=16)
    model = ModelFactory.create_model(
        model_type="linformer",
        input_seq_len=78,
        num_classes=2,
        config=model_config
    )

    # Run benchmarks
    benchmark = ModelBenchmark(
        model=model,
        input_size=(78,),
        device=device
    )

    print(f"\nBenchmarking on device: {device}")
    metrics = benchmark.benchmark()

    print("\nBenchmark Results:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif isinstance(value, (int, float)):
            if "mb" in key.lower():
                print(f"{key}: {value:.2f} MB")
            elif "ms" in key.lower():
                print(f"{key}: {value:.2f} ms")
            else:
                print(f"{key}: {value}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("LINFORMER-IDS ABLATION STUDY EXAMPLES")
    print("="*80)

    # Example 1: Architecture Ablation
    try:
        ablation, results = run_architecture_ablation()
        analyze_results(ablation, results)
    except Exception as e:
        print(f"Example 1 failed: {e}")

    # Example 2: Design Ablation
    try:
        ablation, results = run_design_ablation()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    # Example 3: Comparison
    try:
        compare_multiple_ablations()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    # Example 4: Benchmarking
    try:
        demonstrate_benchmarking()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
