"""Ablation study runner for Linformer-IDS.

This module implements a complete, reproducible ablation study pipeline that:
- Loads base configurations
- Iterates through configured experiments
- Logs results with proper versioning
- Provides reasoning before and after each experiment

Usage:
    python src/run_ablation.py --config configs/ablation/architecture.yaml
    python src/run_ablation.py --config configs/ablation/design.yaml --experiment linformer_k_sweep
    python src/run_ablation.py --config configs/ablation/efficiency.yaml --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_manager import ConfigManager
from src.evaluator import Evaluator
from src.logger import get_logger, setup_logging
from src.model import ModelFactory
from src.trainer import Trainer
from src.utils import create_directory
from pipeline.pipeline import DataPreprocessor
from pipeline.benchmark import FullEfficiencyBenchmark

logger = get_logger(__name__)


class AblationStudy:
    """Manages ablation study execution, logging, and result tracking."""

    def __init__(
        self,
        config_path: str,
        output_base: str = "results",
        logs_base: str = "logs"
    ):
        """Initialize ablation study.

        Args:
            config_path: Path to ablation YAML configuration.
            output_base: Base directory for results.
            logs_base: Base directory for logs.
        """
        self.config_path = Path(config_path)
        self.output_base = Path(output_base)
        self.logs_base = Path(logs_base)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Load ablation config
        with open(self.config_path) as f:
            self.ablation_config = yaml.safe_load(f)

        self.ablation_name = self.ablation_config.get(
            "ablation_name",
            self.config_path.stem
        )

        # Create timestamped directories
        self.experiment_dir = (
            self.output_base /
            "ciciot_binary_classification" /
            self.ablation_name /
            self.timestamp
        )
        self.log_dir = self.logs_base / self.ablation_name / self.timestamp

        create_directory(str(self.experiment_dir))
        create_directory(str(self.log_dir))

        # Setup logging for this experiment
        setup_logging(
            log_dir=str(self.log_dir),
            log_level="INFO",
            console_output=True,
            file_output=True
        )

        logger.info(f"Initialized ablation study: {self.ablation_name}")
        logger.info(f"Results will be saved to: {self.experiment_dir}")
        logger.info(f"Logs will be saved to: {self.log_dir}")

    def load_data(
        self,
        data_path: str,
        preprocessing_config=None,
        task: str = "binary",
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Load and preprocess CIC-IoT data using the same pipeline as main_ciciot.py.

        Steps (matching main_ciciot.py exactly):
          1. encode_features   – label-encode all object columns
          2. normalize         – StandardScaler on numeric features (excl. label)
          3. drop_highly_correlated_features – threshold from config (default 0.9)
          4. drop_low_variance_features      – threshold from config (default 0.0)
          5. drop_columns      – columns_to_drop from config
          6. log1p_transform   – log1p on skewed columns from config

        Args:
            data_path: Path to CIC-IoT CSV file.
            preprocessing_config: PreprocessingConfig object. If None, loaded from
                the base config profile used by this ablation study.

        Returns:
            Tuple of (X, y) arrays (y binary: 0=BenignTraffic, 1=attack).
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Raw DataFrame shape: {df.shape}")

        # Resolve label column case-insensitively
        config_label = preprocessing_config.binary_encode_column if preprocessing_config else "label"
        col_map = {c.lower(): c for c in df.columns}
        label_col = col_map.get(config_label.lower(), config_label)
        if label_col not in df.columns:
            raise ValueError(f"Label column '{config_label}' not found. Columns: {df.columns.tolist()}")

        # Drop rows with missing labels
        df = df.dropna(subset=[label_col])
        logger.info(f"Rows after dropping NaN labels: {df.shape[0]}")

        y_raw = df[label_col].astype(str).values
        unique_labels = np.unique(y_raw)

        if task == "binary":
            benign_labels = [l for l in unique_labels if l.upper() in ('BENIGN', 'BENIGNTRAFFIC', 'NORMAL')]
            if not benign_labels:
                raise ValueError(f"Cannot find benign class in labels: {unique_labels[:10]}")
            benign_label = benign_labels[0]
            logger.info(f"Binary encoding: '{benign_label}'=0, all others=1")
            y = np.where(y_raw == benign_label, 0, 1).astype(np.int64)
            num_classes = 2
        else:
            # Multi-class: integer-encode all labels; BENIGN-like class → 0
            from sklearn.preprocessing import LabelEncoder
            benign_like = [l for l in unique_labels if l.upper() in ('BENIGN', 'BENIGNTRAFFIC', 'NORMAL')]
            benign_label = benign_like[0] if benign_like else unique_labels[0]
            # Sort so benign is index 0
            ordered = [benign_label] + [l for l in sorted(unique_labels) if l != benign_label]
            label_to_int = {l: i for i, l in enumerate(ordered)}
            y = np.array([label_to_int[l] for l in y_raw], dtype=np.int64)
            num_classes = len(ordered)
            logger.info(f"Multi-class encoding: {num_classes} classes, benign=0")
            logger.info(f"Class mapping (first 10): { {k: v for k, v in list(label_to_int.items())[:10]} }")

        # Fill NaN and inf feature values before pipeline (median imputation)
        df_features = df.drop(columns=[label_col])
        # Replace inf/-inf with NaN so median fill catches them
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        nan_count = df_features.isna().sum().sum()
        if nan_count > 0:
            logger.info(f"Filling {nan_count} NaN/inf feature values with column medians")
            df_features = df_features.fillna(df_features.median(numeric_only=True))

        # Run DataPreprocessor pipeline on features only (same 6 steps as main_ciciot.py)
        import copy
        prep_config = copy.deepcopy(preprocessing_config) if preprocessing_config else None
        preprocessor = DataPreprocessor(df_features, config=prep_config)

        logger.info("Step 1/6: Encoding categorical features...")
        preprocessor.encode_features()

        logger.info("Step 2/6: Normalizing features (StandardScaler)...")
        preprocessor.normalize()

        logger.info("Step 3/6: Dropping highly correlated features...")
        preprocessor.drop_highly_correlated_features()

        logger.info("Step 4/6: Dropping low-variance features...")
        preprocessor.drop_low_variance_features()

        logger.info("Step 5/6: Dropping specified columns...")
        preprocessor.drop_columns()

        logger.info("Step 6/6: Applying log1p transformation...")
        preprocessor.log1p_transform()

        X = preprocessor.df.values.astype(np.float32)
        logger.info(f"Processed features shape: {X.shape}")
        logger.info(f"Loaded data: X.shape={X.shape}, y.shape={y.shape}, num_classes={num_classes}")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        return X, y, num_classes

    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for train/val/test splits.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            X_test, y_test: Test data.
            batch_size: Batch size.
            num_workers: Number of workers.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        from torch.utils.data import TensorDataset

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,  # prevents BatchNorm1d failure on single-sample last batch
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader

    def run_experiment(
        self,
        experiment_name: str,
        base_config: Any,
        overrides: Dict[str, Any],
        experiment_config: Dict[str, Any],
        data_loaders: Tuple[DataLoader, DataLoader, DataLoader],
        input_dim: int,
        num_classes: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        run_efficiency: bool = True,
    ) -> Dict[str, Any]:
        """Run a single ablation experiment.

        Args:
            experiment_name: Name of this experiment.
            base_config: Base configuration object.
            overrides: Config overrides for this experiment.
            experiment_config: Experiment metadata.
            data_loaders: Tuple of (train_loader, val_loader, test_loader).
            device: Computation device.

        Returns:
            Dictionary with results and metrics.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment: {experiment_name}")
        logger.info(f"{'='*80}")

        # Log reasoning before experiment
        if "reasoning_before" in experiment_config:
            logger.info("Reasoning BEFORE:")
            for reason in experiment_config["reasoning_before"]:
                logger.info(f"  - {reason}")

        # Separate model_type from config overrides (it is not a config attribute)
        overrides_copy = dict(overrides)
        model_type = overrides_copy.pop("model_type", "linformer")

        config = self._apply_overrides(base_config, overrides_copy)
        logger.info(f"Model type: {model_type}  |  Config overrides: {overrides_copy}")

        # Create model
        try:
            model_config = {
                "dim": config.model.dim,
                "depth": config.model.depth,
                "heads": config.model.heads,
                "k": config.model.k,
                "dropout": config.model.dropout,
                "ff_hidden_mult": config.model.ff_hidden_mult,
                # extra params used by non-Linformer models (ignored if absent)
                "hidden_dims": getattr(config.model, "hidden_dims", [256, 128, 64]),
                "channels": getattr(config.model, "channels", [64, 128, 256]),
                "hidden_size": getattr(config.model, "hidden_size", 128),
                "num_layers": getattr(config.model, "num_layers", 2),
                "use_batchnorm": getattr(config.model, "use_batchnorm", True),
                "bidirectional": getattr(config.model, "bidirectional", True),
            }

            model = ModelFactory.create_model(
                input_seq_len=input_dim,
                num_classes=num_classes,
                model_config=model_config,
                model_type=model_type,
            )
            model.to(device)
            logger.info(f"Model created successfully")
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return {"error": str(e)}

        # Train model — unique checkpoint path per experiment to avoid cross-contamination
        train_loader, val_loader, test_loader = data_loaders
        safe_name = experiment_name.replace("/", "_").replace(" ", "_")
        model_save_path = str(self.experiment_dir / f"checkpoint_{safe_name}.pth")
        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                model_save_path=model_save_path,
            )
            best_model_path = trainer.train()
            logger.info(f"Training completed. Best model: {best_model_path}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}

        # Evaluate
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            # Handle both raw state_dict and checkpoint with metadata
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            evaluator = Evaluator(
                model=model,
                test_loader=test_loader,
                num_classes=num_classes,
                device=device
            )
            metrics = evaluator.evaluate()
            logger.info(f"Evaluation metrics: {metrics}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}

        # Collect training efficiency stats from Trainer
        training_efficiency = {
            "total_training_time_s": round(trainer.total_training_time, 4),
            "epochs_trained": len(trainer.epoch_times),
            "convergence_epoch": trainer.convergence_epoch,
            "avg_epoch_time_s": round(
                sum(trainer.epoch_times) / len(trainer.epoch_times), 4
            ) if trainer.epoch_times else 0.0,
            "min_epoch_time_s": round(min(trainer.epoch_times), 4) if trainer.epoch_times else 0.0,
            "max_epoch_time_s": round(max(trainer.epoch_times), 4) if trainer.epoch_times else 0.0,
            "epoch_times_s": trainer.epoch_times,
        }
        logger.info(f"Training efficiency: {training_efficiency}")

        # Run inference efficiency benchmarks
        efficiency_metrics: Dict[str, Any] = {}
        if run_efficiency:
            try:
                logger.info("Running efficiency benchmarks...")
                bench = FullEfficiencyBenchmark(
                    model=model,
                    model_type=model_type,
                    input_dim=input_dim,
                    device=device,
                    checkpoint_path=model_save_path,
                )
                efficiency_metrics = bench.run_full_benchmark()
                logger.info("Efficiency benchmarks completed.")
            except Exception as e:
                logger.warning(f"Efficiency benchmarks failed: {e}")
                efficiency_metrics = {"efficiency_error": str(e)}

        # Save results
        result = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "num_classes": num_classes,
            "config_overrides": overrides,
            "metrics": metrics,
            "training_efficiency": training_efficiency,
            "efficiency_benchmark": efficiency_metrics,
        }

        # Log reasoning after experiment
        if "reasoning_after" in experiment_config:
            logger.info("Reasoning AFTER:")
            result["reasoning_after"] = []
            for reason in experiment_config["reasoning_after"]:
                logger.info(f"  - {reason}")
                result["reasoning_after"].append(reason)

        return result

    def run_ablation_study(
        self,
        data_path: str,
        experiment_filter: Optional[str] = None,
        device: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Run complete ablation study.

        Args:
            data_path: Path to training data.
            experiment_filter: Run only experiments matching this filter.
            device: Device to use ('cuda' or 'cpu').

        Returns:
            List of results from all experiments.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load base configuration
        base_profile = self.ablation_config.get("base_profile", "first")
        logger.info(f"Loading base configuration profile: {base_profile}")

        # Construct path to config (handle relative paths)
        config_path = Path("configs/config.yaml")
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

        config = ConfigManager.load_config(
            str(config_path),
            profile=base_profile
        )

        # Apply training overrides if specified
        if "training_overrides" in self.ablation_config:
            logger.info("Applying training overrides from ablation config")
            overrides = self.ablation_config["training_overrides"]
            for key, value in overrides.items():
                setattr(config.training, key, value)
                logger.info(f"  training.{key} = {value}")

        # Load data using the full preprocessing pipeline from config
        task = self.ablation_config.get("task", "binary")
        X, y, num_classes = self.load_data(
            data_path, preprocessing_config=config.preprocessing, task=task
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )

        # Create data loaders
        data_loaders = self.create_data_loaders(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            batch_size=config.training.batch_size
        )

        # Get input dimension
        input_dim = X_train.shape[1]
        logger.info(f"Input dimension: {input_dim}")

        # Run experiments
        results = []
        experiments = self.ablation_config.get("experiments", [])

        for exp_config in experiments:
            exp_name = exp_config.get("name")

            # Filter experiments if requested
            if experiment_filter and exp_name != experiment_filter:
                logger.info(f"Skipping {exp_name} (not matching filter)")
                continue

            # Handle different variable types
            if "variables" in exp_config:
                # Simple list of values for one variable
                var_name = list(exp_config["variables"].keys())[0]
                var_values = exp_config["variables"][var_name]

                for value in var_values:
                    run_name = f"{exp_name}_{var_name}_{value}"
                    overrides = {f"model.{var_name}": value}

                    result = self.run_experiment(
                        run_name,
                        config,
                        overrides,
                        exp_config,
                        data_loaders,
                        input_dim,
                        num_classes=num_classes,
                        device=device,
                    )
                    results.append(result)

            elif "model_types" in exp_config:
                # Compare multiple model architectures
                for model_entry in exp_config["model_types"]:
                    mt = model_entry["type"]
                    run_name = f"{exp_name}_{mt}"
                    overrides = {"model_type": mt}
                    # Forward any per-model arch params (e.g. hidden_dims, channels) as model.* overrides
                    for k, v in model_entry.items():
                        if k != "type":
                            overrides[f"model.{k}"] = v
                    result = self.run_experiment(
                        run_name,
                        config,
                        overrides,
                        exp_config,
                        data_loaders,
                        input_dim,
                        num_classes=num_classes,
                        device=device,
                    )
                    results.append(result)

            elif "variables_grid" in exp_config:
                # Grid search across multiple variables
                grid = exp_config["variables_grid"][0]
                keys = list(grid.keys())
                values = [grid[k] for k in keys]

                for combo in product(*values):
                    overrides = {f"model.{k}": v for k, v in zip(keys, combo)}
                    run_name = f"{exp_name}_" + "_".join(
                        f"{k}_{v}" for k, v in zip(keys, combo)
                    )

                    result = self.run_experiment(
                        run_name,
                        config,
                        overrides,
                        exp_config,
                        data_loaders,
                        input_dim,
                        num_classes=num_classes,
                        device=device,
                    )
                    results.append(result)

        # Save all results
        self._save_results(results)
        return results

    def _apply_overrides(self, base_config: Any, overrides: Dict[str, Any]) -> Any:
        """Apply config overrides to base configuration.

        Args:
            base_config: Base configuration object.
            overrides: Dictionary of overrides (dot-notation keys).

        Returns:
            Modified configuration object.
        """
        import copy
        config = copy.deepcopy(base_config)

        for key, value in overrides.items():
            parts = key.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        return config

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save experiment results to files.

        Args:
            results: List of experiment results.
        """
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return super().default(obj)

        # Save as JSON
        results_file = self.experiment_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
        logger.info(f"Saved results to {results_file}")

        # Save as CSV (for easier analysis)
        csv_file = self.experiment_dir / "results.csv"
        rows = []
        for result in results:
            if "error" in result:
                continue
            row = {
                "experiment": result["experiment_name"],
                "model_type": result.get("model_type", ""),
                "num_classes": result.get("num_classes", 2),
            }
            row.update(result.get("config_overrides", {}))

            # Accuracy/classification metrics (scalars only)
            metrics = result.get("metrics", {})
            scalar_metrics = metrics.get("metrics", metrics)
            if isinstance(scalar_metrics, dict):
                row.update({k: v for k, v in scalar_metrics.items()
                            if isinstance(v, (int, float, np.integer, np.floating))})

            # Training efficiency (scalars only)
            te = result.get("training_efficiency", {})
            for k in ("total_training_time_s", "avg_epoch_time_s", "epochs_trained",
                      "convergence_epoch", "min_epoch_time_s", "max_epoch_time_s"):
                if k in te:
                    row[f"train_{k}"] = te[k]

            # Key inference efficiency metrics (scalars only, flat dict)
            eb = result.get("efficiency_benchmark", {})
            keep_keys = (
                "total_parameters", "model_size_mb", "flops_per_sample", "gflops_per_sample",
                "latency_mean_ms", "latency_std_ms", "latency_p50_ms", "latency_p95_ms",
                "latency_p99_ms", "latency_jitter_p99_p50_ms",
                "throughput_bs1_samples_per_sec", "throughput_bs128_samples_per_sec",
                "ram_delta_mb", "gpu_peak_memory_mb",
                "disk_size_mb", "disk_size_mb_estimated",
                "cold_start_mean_ms", "cpu_gpu_speedup_ratio",
                "complexity_class",
                "energy_per_inference_uj", "energy_device_tdp_watts",
                "edge_tier", "edge_feasible",
                "load_bs1024_throughput_samples_per_sec",
                "latency_cpu_mean_ms", "latency_cuda_mean_ms",
            )
            for k in keep_keys:
                if k in eb:
                    row[f"bench_{k}"] = eb[k]

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved results CSV to {csv_file}")

        # Save summary
        summary = {
            "ablation_name": self.ablation_name,
            "timestamp": self.timestamp,
            "total_experiments": len(results),
            "successful_experiments": sum(1 for r in results if "error" not in r),
            "results_file": str(results_file),
            "csv_file": str(csv_file)
        }

        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")


def main():
    """Parse arguments and run ablation study."""
    parser = argparse.ArgumentParser(
        description="Run ablation study for Linformer-IDS"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ablation configuration YAML"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="lightweight-transformer-ids/data/raw/ciciot/ciciot_training_50_50.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run only a specific experiment (by name)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu')"
    )

    args = parser.parse_args()

    # Run ablation study
    ablation = AblationStudy(args.config)
    results = ablation.run_ablation_study(
        data_path=args.data,
        experiment_filter=args.experiment,
        device=args.device
    )

    logger.info(f"\nAblation study completed!")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Results saved to: {ablation.experiment_dir}")

    return results


if __name__ == "__main__":
    main()
