"""Efficiency benchmarking for Linformer-IDS models.

This module provides utilities to measure model efficiency metrics:
- Parameter count and model size
- FLOPs (floating point operations)
- Latency and throughput
- Peak memory usage
- Device-specific performance

Usage:
    from pipeline.benchmark import ModelBenchmark
    benchmark = ModelBenchmark(model, input_size=(78,), device='cuda')
    metrics = benchmark.benchmark()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from fvcore.nn import FlopCounterMode
    FLOPS_AVAILABLE = True
except ImportError:
    FLOPS_AVAILABLE = False


class ModelBenchmark:
    """Benchmark model efficiency metrics."""

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...],
        device: str = "cpu",
        batch_sizes: Optional[list] = None
    ):
        """Initialize benchmarking.

        Args:
            model: PyTorch model to benchmark.
            input_size: Input size (e.g., (78,) for CIC-IoT features).
            device: Device to benchmark on ('cuda' or 'cpu').
            batch_sizes: Batch sizes to test (default: [1, 8, 32, 128]).
        """
        self.model = model.to(device)
        self.input_size = input_size
        self.device = device
        self.batch_sizes = batch_sizes or [1, 8, 32, 128]
        self.model.eval()

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with parameter counts.
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = total - trainable

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "model_size_mb": total * 4 / (1024 * 1024)  # Assuming float32
        }

    def estimate_flops(self) -> Dict[str, Any]:
        """Estimate FLOPs using fvcore or manual calculation.

        Returns:
            Dictionary with FLOP estimates.
        """
        if not FLOPS_AVAILABLE:
            return {"flops_error": "fvcore not installed"}

        try:
            # Create dummy input
            dummy_input = torch.randn(
                1, *self.input_size,
                device=self.device
            )

            # Count FLOPs
            with torch.no_grad():
                with FlopCounterMode(self.model) as flop_counter:
                    _ = self.model(dummy_input)

                flops = flop_counter.total()

            return {
                "total_flops": flops,
                "flops_per_input": flops
            }
        except Exception as e:
            return {"flops_error": str(e)}

    def measure_latency(
        self,
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ) -> Dict[str, Any]:
        """Measure inference latency.

        Args:
            warmup_runs: Number of warmup runs.
            benchmark_runs: Number of benchmark runs.

        Returns:
            Dictionary with latency statistics (ms).
        """
        dummy_input = torch.randn(
            1, *self.input_size,
            device=self.device
        )

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(benchmark_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.model(dummy_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)
        return {
            "latency_ms_mean": float(np.mean(latencies)),
            "latency_ms_std": float(np.std(latencies)),
            "latency_ms_min": float(np.min(latencies)),
            "latency_ms_max": float(np.max(latencies)),
            "latency_ms_p50": float(np.percentile(latencies, 50)),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
            "latency_ms_p99": float(np.percentile(latencies, 99))
        }

    def measure_throughput(
        self,
        duration_seconds: int = 10
    ) -> Dict[str, float]:
        """Measure inference throughput.

        Args:
            duration_seconds: Benchmark duration.

        Returns:
            Dictionary with throughput metrics.
        """
        throughputs = {}

        for batch_size in self.batch_sizes:
            dummy_input = torch.randn(
                batch_size, *self.input_size,
                device=self.device
            )

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_input)

            if self.device == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            sample_count = 0

            with torch.no_grad():
                while time.perf_counter() - start < duration_seconds:
                    _ = self.model(dummy_input)
                    sample_count += batch_size

            if self.device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            throughput = sample_count / elapsed

            throughputs[f"throughput_bs{batch_size}_samples_per_sec"] = throughput

        return throughputs

    def measure_memory(self) -> Dict[str, float]:
        """Measure peak memory usage during inference.

        Returns:
            Dictionary with memory statistics (MB).
        """
        if self.device != "cuda":
            return {"memory_error": "Memory measurement only available on CUDA"}

        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            dummy_input = torch.randn(
                1, *self.input_size,
                device=self.device
            )

            with torch.no_grad():
                _ = self.model(dummy_input)

            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return {"peak_memory_mb": float(peak_memory)}
        except Exception as e:
            return {"memory_error": str(e)}

    def benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite.

        Returns:
            Dictionary with all benchmark metrics.
        """
        metrics = {}

        # Parameter count
        metrics.update(self.count_parameters())

        # FLOPs
        metrics.update(self.estimate_flops())

        # Latency
        metrics.update(self.measure_latency())

        # Throughput
        metrics.update(self.measure_throughput())

        # Memory (CUDA only)
        if self.device == "cuda":
            metrics.update(self.measure_memory())

        return metrics

    def benchmark_batch_sensitivity(self) -> Dict[str, Any]:
        """Detailed batch size sensitivity analysis.

        Returns:
            Dictionary with metrics across batch sizes.
        """
        results = {"batch_sizes": {}}

        for batch_size in self.batch_sizes:
            dummy_input = torch.randn(
                batch_size, *self.input_size,
                device=self.device
            )

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_input)

            # Measure latency
            latencies = []
            with torch.no_grad():
                for _ in range(50):
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = self.model(dummy_input)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)

            latency_total_ms = np.mean(latencies)
            latency_per_sample_ms = latency_total_ms / batch_size
            throughput = (batch_size * 1000) / latency_total_ms

            results["batch_sizes"][f"bs_{batch_size}"] = {
                "latency_total_ms": float(latency_total_ms),
                "latency_per_sample_ms": float(latency_per_sample_ms),
                "throughput_samples_per_sec": float(throughput)
            }

        return results


class ComparativeBenchmark:
    """Compare efficiency across multiple models."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize comparative benchmarking.

        Args:
            device: Device to benchmark on.
        """
        self.device = device
        self.results = []

    def add_model(
        self,
        name: str,
        model: nn.Module,
        input_size: Tuple[int, ...]
    ) -> None:
        """Add a model for benchmarking.

        Args:
            name: Model identifier.
            model: PyTorch model.
            input_size: Input size.
        """
        benchmark = ModelBenchmark(model, input_size, self.device)
        metrics = benchmark.benchmark()
        metrics["model_name"] = name

        self.results.append(metrics)

    def summary_dataframe(self):
        """Get benchmark results as DataFrame.

        Returns:
            pandas.DataFrame with benchmark results.
        """
        try:
            import pandas as pd
            df = pd.DataFrame(self.results)
            return df
        except ImportError:
            return None

    def print_summary(self) -> None:
        """Print benchmark comparison summary."""
        print("\n" + "="*80)
        print("COMPARATIVE BENCHMARK RESULTS")
        print("="*80)

        for result in self.results:
            print(f"\nModel: {result.get('model_name', 'Unknown')}")
            print(f"  Parameters: {result.get('total_parameters', 'N/A'):,}")
            print(f"  Model Size: {result.get('model_size_mb', 'N/A'):.2f} MB")
            print(f"  Latency: {result.get('latency_ms_mean', 'N/A'):.2f}ms (mean)")
            print(f"  FLOPs: {result.get('total_flops', 'N/A'):,}")

        print("\n" + "="*80)


def benchmark_architecture_comparison(
    models_config: Dict[str, Dict[str, Any]],
    input_size: Tuple[int, ...] = (78,),
    device: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Benchmark multiple model architectures for comparison.

    Args:
        models_config: Dictionary of model names to config dicts.
        input_size: Input size for models.
        device: Device to benchmark on.

    Returns:
        Dictionary with benchmark results for each model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    for model_name, config in models_config.items():
        try:
            from src.model import ModelFactory
            model = ModelFactory.create_model(
                model_type=config.get("type", "linformer"),
                input_seq_len=input_size[0],
                num_classes=2,
                config=config.get("model_config")
            )

            benchmark = ModelBenchmark(model, input_size, device)
            results[model_name] = benchmark.benchmark()
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Comprehensive efficiency suite covering all requested metric categories
# ---------------------------------------------------------------------------

class FullEfficiencyBenchmark:
    """Complete efficiency benchmark covering all metric categories.

    Categories:
      - Core: params, FLOPs, latency, throughput, RAM/GPU memory
      - Training: epoch time, total time, convergence speed (injected externally)
      - Scalability: complexity class, memory/latency vs input size, batch scaling
      - Deployment: disk size, cold start, CPU/GPU gap, edge feasibility
      - Stability: latency variance (p50/p95/p99), load test, constrained memory
    """

    # Theoretical complexity per model type
    COMPLEXITY_MAP = {
        "mlp":         "O(n)",      # n = input_dim; one pass through layers
        "cnn":         "O(n)",      # 1-D conv, linear in n
        "lstm":        "O(n)",      # n time steps, linear scan
        "transformer": "O(n^2)",    # full self-attention
        "linformer":   "O(nk)",     # low-rank projection (k << n)
    }

    # Approximate GPU TDP (W) for common GPUs; used for energy estimation
    _GPU_TDP = {
        "rtx 5090": 575,
        "rtx 4090": 450,
        "rtx 3090": 350,
        "rtx 3080": 320,
        "a100":     400,
        "default":  250,
    }

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        input_dim: int,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        warmup: int = 20,
        runs: int = 200,
    ) -> None:
        self.model = model.eval()
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.warmup = warmup
        self.runs = runs
        self._dev = torch.device(device)
        self.model.to(self._dev)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dummy(self, batch: int = 1, seq: Optional[int] = None) -> torch.Tensor:
        s = seq if seq is not None else self.input_dim
        return torch.randn(batch, s, device=self._dev)

    def _sync(self) -> None:
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _gpu_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0).lower()
        return ""

    def _gpu_tdp_watts(self) -> float:
        name = self._gpu_name()
        for key, tdp in self._GPU_TDP.items():
            if key in name:
                return float(tdp)
        return float(self._GPU_TDP["default"])

    # ------------------------------------------------------------------
    # 1. Core metrics
    # ------------------------------------------------------------------

    def measure_parameters(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        size_mb = total * 4 / (1024 ** 2)  # float32
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "model_size_bytes_float32": total * 4,
            "model_size_mb": round(size_mb, 4),
        }

    def measure_flops(self) -> Dict[str, Any]:
        if not FLOPS_AVAILABLE:
            return {"flops_note": "fvcore not installed; skipped"}
        try:
            x = self._dummy(1)
            with torch.no_grad():
                with FlopCounterMode(self.model) as fc:
                    self.model(x)
            flops = fc.total()
            return {"flops_per_sample": int(flops), "gflops_per_sample": round(flops / 1e9, 6)}
        except Exception as e:
            return {"flops_note": f"FLOPs estimation failed: {e}"}

    def measure_latency(self) -> Dict[str, float]:
        x = self._dummy(1)
        with torch.no_grad():
            for _ in range(self.warmup):
                self.model(x)
        self._sync()

        latencies_ms: List[float] = []
        with torch.no_grad():
            for _ in range(self.runs):
                self._sync()
                t0 = time.perf_counter()
                self.model(x)
                self._sync()
                latencies_ms.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies_ms)
        return {
            "latency_mean_ms":  round(float(np.mean(arr)), 4),
            "latency_std_ms":   round(float(np.std(arr)), 4),
            "latency_min_ms":   round(float(np.min(arr)), 4),
            "latency_max_ms":   round(float(np.max(arr)), 4),
            "latency_p50_ms":   round(float(np.percentile(arr, 50)), 4),
            "latency_p95_ms":   round(float(np.percentile(arr, 95)), 4),
            "latency_p99_ms":   round(float(np.percentile(arr, 99)), 4),
            "latency_jitter_p99_p50_ms": round(
                float(np.percentile(arr, 99) - np.percentile(arr, 50)), 4
            ),
        }

    def measure_throughput(self, batch_sizes: List[int] = None) -> Dict[str, float]:
        batch_sizes = batch_sizes or [1, 32, 128, 512]
        results: Dict[str, float] = {}
        for bs in batch_sizes:
            x = self._dummy(bs)
            with torch.no_grad():
                for _ in range(5):
                    self.model(x)
            self._sync()

            latencies: List[float] = []
            with torch.no_grad():
                for _ in range(50):
                    self._sync()
                    t0 = time.perf_counter()
                    self.model(x)
                    self._sync()
                    latencies.append(time.perf_counter() - t0)

            mean_sec = float(np.mean(latencies))
            results[f"throughput_bs{bs}_samples_per_sec"] = round(bs / mean_sec, 2)
            results[f"latency_bs{bs}_mean_ms"] = round(mean_sec * 1000, 4)
            results[f"latency_per_sample_bs{bs}_ms"] = round(mean_sec * 1000 / bs, 6)
        return results

    def measure_ram_usage(self) -> Dict[str, float]:
        """Peak host RAM during a single forward pass (via psutil RSS delta)."""
        if not PSUTIL_AVAILABLE:
            return {"ram_note": "psutil not installed; skipped"}
        proc = psutil.Process(os.getpid())
        # Force GC baseline
        import gc; gc.collect()
        base_mb = proc.memory_info().rss / (1024 ** 2)
        x = self._dummy(1)
        with torch.no_grad():
            self.model(x)
        peak_mb = proc.memory_info().rss / (1024 ** 2)
        return {
            "ram_baseline_mb": round(base_mb, 2),
            "ram_after_inference_mb": round(peak_mb, 2),
            "ram_delta_mb": round(peak_mb - base_mb, 2),
        }

    def measure_gpu_memory(self) -> Dict[str, float]:
        """Peak GPU VRAM during a single-sample inference."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return {"gpu_memory_note": "CUDA not available; skipped"}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        x = self._dummy(1)
        with torch.no_grad():
            self.model(x)
        self._sync()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return {
            "gpu_peak_memory_mb": round(peak_mb, 4),
            "gpu_peak_memory_kb": round(peak_mb * 1024, 2),
        }

    # ------------------------------------------------------------------
    # 2. Deployment metrics
    # ------------------------------------------------------------------

    def measure_disk_size(self) -> Dict[str, Any]:
        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
            size_bytes = os.path.getsize(self.checkpoint_path)
            return {
                "disk_size_bytes": size_bytes,
                "disk_size_mb": round(size_bytes / (1024 ** 2), 4),
            }
        # Estimate from parameters (float32)
        params = sum(p.numel() for p in self.model.parameters())
        estimated_mb = params * 4 / (1024 ** 2)
        return {
            "disk_size_mb_estimated": round(estimated_mb, 4),
            "disk_size_note": "checkpoint path not provided; estimated from params",
        }

    def measure_cold_start(self, n_trials: int = 5) -> Dict[str, float]:
        """Time to load model from checkpoint + run first inference."""
        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            return {"cold_start_note": "checkpoint path not provided; skipped"}
        import copy
        times_ms: List[float] = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            # Reload state dict from disk
            ckpt = torch.load(self.checkpoint_path, map_location=self._dev)
            m2 = copy.deepcopy(self.model)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                m2.load_state_dict(ckpt["model_state_dict"])
            else:
                m2.load_state_dict(ckpt)
            m2.eval()
            x = self._dummy(1)
            with torch.no_grad():
                m2(x)
            self._sync()
            times_ms.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times_ms)
        return {
            "cold_start_mean_ms": round(float(np.mean(arr)), 2),
            "cold_start_std_ms":  round(float(np.std(arr)), 2),
            "cold_start_min_ms":  round(float(np.min(arr)), 2),
        }

    def measure_cpu_gpu_gap(self) -> Dict[str, Any]:
        """Compare latency on CPU vs GPU (single sample)."""
        results: Dict[str, Any] = {}
        for dev_name in ["cpu", "cuda"]:
            if dev_name == "cuda" and not torch.cuda.is_available():
                continue
            dev = torch.device(dev_name)
            import copy
            m = copy.deepcopy(self.model).to(dev).eval()
            x = torch.randn(1, self.input_dim, device=dev)
            with torch.no_grad():
                for _ in range(10):
                    m(x)
            if dev_name == "cuda":
                torch.cuda.synchronize()
            lat: List[float] = []
            with torch.no_grad():
                for _ in range(100):
                    if dev_name == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    m(x)
                    if dev_name == "cuda":
                        torch.cuda.synchronize()
                    lat.append((time.perf_counter() - t0) * 1000)
            results[f"latency_{dev_name}_mean_ms"] = round(float(np.mean(lat)), 4)
            results[f"latency_{dev_name}_p95_ms"] = round(float(np.percentile(lat, 95)), 4)
        if "latency_cpu_mean_ms" in results and "latency_cuda_mean_ms" in results:
            ratio = results["latency_cpu_mean_ms"] / results["latency_cuda_mean_ms"]
            results["cpu_gpu_speedup_ratio"] = round(ratio, 2)
        return results

    def assess_edge_deployability(self) -> Dict[str, Any]:
        """Heuristic edge deployability assessment."""
        params = self.measure_parameters()
        n = params["total_parameters"]
        mem_mb = params["model_size_mb"]
        lat = self.measure_latency()
        mean_lat = lat["latency_mean_ms"]

        # Thresholds: Raspberry Pi class device targets
        feasible = n < 5_000_000 and mem_mb < 20 and mean_lat < 50
        tier = (
            "edge-ready" if feasible else
            "constrained-edge" if (n < 20_000_000 and mem_mb < 80) else
            "cloud/server-only"
        )
        return {
            "edge_tier": tier,
            "edge_feasible": feasible,
            "edge_note": (
                f"params={n:,}, size={mem_mb:.2f}MB, latency={mean_lat:.2f}ms — "
                f"{'within' if feasible else 'exceeds'} edge constraints"
            ),
        }

    # ------------------------------------------------------------------
    # 3. Scalability metrics
    # ------------------------------------------------------------------

    def measure_complexity_class(self) -> Dict[str, str]:
        return {
            "complexity_class": self.COMPLEXITY_MAP.get(self.model_type, "unknown"),
            "complexity_note": {
                "mlp":         "O(n) — linear in feature count, no cross-feature attention",
                "cnn":         "O(n) — linear scan with fixed-size kernels",
                "lstm":        "O(n) — sequential recurrence over n time steps",
                "transformer": "O(n^2) — full pairwise self-attention matrix",
                "linformer":   "O(nk) — projected attention, k << n",
            }.get(self.model_type, ""),
        }

    def measure_latency_vs_input_size(
        self, sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Latency at different input sequence lengths (scalability)."""
        sizes = sizes or [8, 16, 25, 32, 64, 128]
        results: Dict[str, Any] = {}
        for s in sizes:
            x = torch.randn(1, s, device=self._dev)
            try:
                with torch.no_grad():
                    for _ in range(5):
                        self.model(x)
                self._sync()
                lat: List[float] = []
                with torch.no_grad():
                    for _ in range(50):
                        self._sync()
                        t0 = time.perf_counter()
                        self.model(x)
                        self._sync()
                        lat.append((time.perf_counter() - t0) * 1000)
                results[f"latency_seqlen{s}_ms"] = round(float(np.mean(lat)), 4)
            except Exception as e:
                results[f"latency_seqlen{s}_ms"] = f"error: {e}"
        return results

    def measure_gpu_memory_vs_input_size(
        self, sizes: List[int] = None
    ) -> Dict[str, Any]:
        """GPU VRAM at different input sequence lengths."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return {"gpu_scaling_note": "CUDA not available; skipped"}
        sizes = sizes or [8, 16, 25, 32, 64, 128]
        results: Dict[str, Any] = {}
        for s in sizes:
            x = torch.randn(1, s, device=self._dev)
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                with torch.no_grad():
                    self.model(x)
                self._sync()
                mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                results[f"gpu_memory_seqlen{s}_mb"] = round(mb, 4)
            except Exception as e:
                results[f"gpu_memory_seqlen{s}_mb"] = f"error: {e}"
        return results

    # ------------------------------------------------------------------
    # 4. Stability & practical metrics
    # ------------------------------------------------------------------

    def measure_load_test(
        self, load_batch_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Stress test with large batches to simulate high traffic."""
        load_batch_sizes = load_batch_sizes or [256, 512, 1024, 4096]
        results: Dict[str, Any] = {}
        for bs in load_batch_sizes:
            try:
                x = self._dummy(bs)
                with torch.no_grad():
                    for _ in range(3):
                        self.model(x)
                self._sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    self.model(x)
                self._sync()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                results[f"load_bs{bs}_latency_ms"] = round(elapsed_ms, 4)
                results[f"load_bs{bs}_throughput_samples_per_sec"] = round(
                    bs / (elapsed_ms / 1000), 2
                )
                if self.device == "cuda":
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    results[f"load_bs{bs}_gpu_peak_mb"] = round(peak_mb, 2)
            except RuntimeError as e:
                results[f"load_bs{bs}_error"] = str(e)
        return results

    def measure_energy_estimate(self) -> Dict[str, Any]:
        """Estimate energy per inference from latency × device TDP."""
        lat = self.measure_latency()
        mean_s = lat["latency_mean_ms"] / 1000.0
        if self.device == "cuda":
            tdp_w = self._gpu_tdp_watts()
            device_label = f"GPU ({self._gpu_name() or 'unknown'})"
        else:
            tdp_w = 65.0   # typical laptop/desktop CPU
            device_label = "CPU"
        energy_uj = mean_s * tdp_w * 1e6  # micro-joules
        return {
            "energy_device": device_label,
            "energy_device_tdp_watts": tdp_w,
            "energy_per_inference_us": round(mean_s * 1e6, 4),  # latency in µs
            "energy_per_inference_uj": round(energy_uj, 4),
            "energy_per_1k_inferences_mj": round(energy_uj * 1000 / 1000, 4),
            "energy_note": "estimated from latency × device TDP; not measured directly",
        }

    # ------------------------------------------------------------------
    # 5. Full suite runner
    # ------------------------------------------------------------------

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run every metric category and return a flat results dict."""
        results: Dict[str, Any] = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "device": self.device,
        }

        print(f"  [params]        ", end="", flush=True)
        results.update(self.measure_parameters())

        print(f"FLOPs ", end="", flush=True)
        results.update(self.measure_flops())

        print(f"latency ", end="", flush=True)
        results.update(self.measure_latency())

        print(f"throughput ", end="", flush=True)
        results.update(self.measure_throughput())

        print(f"RAM ", end="", flush=True)
        results.update(self.measure_ram_usage())

        if self.device == "cuda":
            print(f"GPU-mem ", end="", flush=True)
            results.update(self.measure_gpu_memory())

        print(f"disk ", end="", flush=True)
        results.update(self.measure_disk_size())

        print(f"cold-start ", end="", flush=True)
        results.update(self.measure_cold_start())

        print(f"cpu/gpu-gap ", end="", flush=True)
        results.update(self.measure_cpu_gpu_gap())

        print(f"complexity ", end="", flush=True)
        results.update(self.measure_complexity_class())

        print(f"scaling ", end="", flush=True)
        results.update(self.measure_latency_vs_input_size())
        if self.device == "cuda":
            results.update(self.measure_gpu_memory_vs_input_size())

        print(f"batch-scaling ", end="", flush=True)
        results.update(self.measure_throughput([1, 8, 32, 128, 512]))

        print(f"load-test ", end="", flush=True)
        results.update(self.measure_load_test())

        print(f"energy ", end="", flush=True)
        results.update(self.measure_energy_estimate())

        print(f"edge ", end="", flush=True)
        results.update(self.assess_edge_deployability())

        print("done")
        return results
