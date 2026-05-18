"""8-Class Classification Ablation Study (7 attack classes + Benign).

Trains MLP, CNN, LSTM, Transformer, and Linformer on a merged 8-class CIC-IoT
dataset and evaluates on all available test sets.

Classes:
    0  Benign      1  DDoS       2  DoS        3  Spoofing
    4  Recon       5  Web-based  6  Brute Force 7  Mirai

Data strategy:
    Training : ciciot_training_50_50 (attacks + benign) + 80% zero_day → merged in memory
    Test sets : ciciot_test_5_95, ciciot_test_1_99,
                ciciot_test_99_1 (sampled), zero_day 20% holdout

Usage:
    python src/run_7class_ablation.py \\
        --config configs/ablation/8class_comparison.yaml \\
        --data-root data/raw/ciciot_2023 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_manager import ConfigManager
from src.evaluator import Evaluator
from src.logger import get_logger, setup_logging
from src.model import ModelFactory
from src.trainer import Trainer
from src.utils import create_directory
from pipeline.benchmark import FullEfficiencyBenchmark

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 8-class constants (7 attack types + Benign)
# ──────────────────────────────────────────────────────────────────────────────

COMMON_FEATURES = [
    "ARP", "AVG", "DHCP", "DNS", "HTTP", "HTTPS", "Header_Length", "IAT",
    "ICMP", "IPv", "IRC", "LLC", "Max", "Min", "Number", "Protocol Type",
    "Rate", "SMTP", "SSH", "Std", "TCP", "Telnet", "Tot size", "Tot sum",
    "UDP", "Variance", "ack_count", "ack_flag_number", "cwr_flag_number",
    "ece_flag_number", "fin_count", "fin_flag_number", "psh_flag_number",
    "rst_count", "rst_flag_number", "syn_count", "syn_flag_number",
]

CLASS_NAMES = ["Benign", "DDoS", "DoS", "Spoofing", "Recon", "Web-based", "Brute Force", "Mirai"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES   = len(CLASS_NAMES)

LABEL_MAP: Dict[str, str] = {
    # Benign traffic
    "BENIGN": "Benign", "BENIGNTRAFFIC": "Benign", "NORMAL": "Benign",
    # DDoS
    "DDOS-ACK_FRAGMENTATION": "DDoS", "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-SLOWLORIS": "DDoS", "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS", "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS", "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS", "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS", "DDOS-ICMP_FRAGMENTATION": "DDoS",
    # DoS
    "DOS-TCP_FLOOD": "DoS", "DOS-HTTP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS", "DOS-UDP_FLOOD": "DoS",
    # Spoofing
    "MITM-ARPSPOOFING": "Spoofing", "DNS_SPOOFING": "Spoofing",
    # Recon
    "RECONNAISSANCE-PINGSWEEP": "Recon", "RECONNAISSANCE-OSSCAN": "Recon",
    "VULNERABILITYSCAN": "Recon", "RECONNAISSANCE-PORTSCAN": "Recon",
    "RECONNAISSANCE-HOSTDISCOVERY": "Recon", "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon", "RECON-PORTSCAN": "Recon", "RECON-PINGSWEEP": "Recon",
    # Web-based
    "SQLINJECTION": "Web-based", "COMMANDINJECTION": "Web-based",
    "BACKDOOR_MALWARE": "Web-based", "UPLOADING_ATTACK": "Web-based",
    "XSS": "Web-based", "BROWSERHIJACKING": "Web-based",
    # Brute Force
    "DICTIONARYBRUTEFORCE": "Brute Force",
    # Mirai
    "MIRAI-GREIP_FLOOD": "Mirai", "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_and_map(
    csv_path: str,
    drop_benign: bool = False,
    sample_n: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV → extract 37 common features → map labels to 8 classes.

    Args:
        csv_path: Path to raw CSV (never modified).
        drop_benign: Drop BENIGN rows (set True only if benign is unwanted).
        sample_n: If set, randomly sample this many rows before processing.

    Returns:
        X (n, 37) float32, y (n,) int64
    """
    logger.info(f"  Reading {Path(csv_path).name} ...")
    df = pd.read_csv(csv_path)

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        logger.info(f"  Sampled {sample_n:,} rows from {Path(csv_path).name}")

    lc = next(c for c in df.columns if c.lower() == "label")

    if drop_benign:
        benign_mask = df[lc].astype(str).str.upper().isin(
            {"BENIGN", "BENIGNTRAFFIC", "NORMAL"}
        )
        df = df[~benign_mask].reset_index(drop=True)

    mapped = df[lc].astype(str).str.strip().str.upper().map(LABEL_MAP)
    valid  = mapped.notna()
    df     = df[valid].reset_index(drop=True)
    mapped = mapped[valid].reset_index(drop=True)

    X_df = pd.DataFrame(index=df.index)
    for col in COMMON_FEATURES:
        X_df[col] = df[col] if col in df.columns else 0.0

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    X = X_df.values.astype(np.float32)
    y = np.array([CLASS_TO_IDX[c] for c in mapped], dtype=np.int64)

    dist = {CLASS_NAMES[i]: int((y == i).sum())
            for i in range(NUM_CLASSES) if (y == i).sum() > 0}
    logger.info(f"  Loaded {len(X):,} rows → {dist}")
    return X, y


def oversample_minority(
    X: np.ndarray, y: np.ndarray, target: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Random oversampling of minority classes to `target` samples (in memory)."""
    rng = np.random.default_rng(seed)
    Xp, yp = [X], [y]
    for cls in range(NUM_CLASSES):
        mask = y == cls
        n = mask.sum()
        if 0 < n < target:
            idx = rng.choice(np.where(mask)[0], target - n, replace=True)
            Xp.append(X[idx])
            yp.append(y[idx])
    return np.vstack(Xp), np.concatenate(yp)


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=shuffle)


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    w = 1.0 / counts
    w = w / w.sum() * NUM_CLASSES
    return torch.FloatTensor(w).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    checkpoint_path: str,
    cw: torch.Tensor,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_macro  = 0.0
    no_improve  = 0
    history: Dict[str, Any] = {
        "train_loss": [], "val_macro_f1": [], "val_weighted_f1": [],
        "best_epoch": 0, "epoch_times": [],
    }
    t_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        total_loss, n = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
        scheduler.step()

        train_loss = total_loss / n
        history["train_loss"].append(round(train_loss, 6))

        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds = model(Xb.to(device)).argmax(1).cpu().numpy()
                all_true.extend(yb.numpy())
                all_pred.extend(preds)
        macro_f1    = f1_score(all_true, all_pred, average="macro",    zero_division=0)
        weighted_f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0)
        history["val_macro_f1"].append(round(macro_f1, 6))
        history["val_weighted_f1"].append(round(weighted_f1, 6))
        ep_time = time.perf_counter() - t_ep
        history["epoch_times"].append(round(ep_time, 3))

        logger.info(f"    Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
                    f"macro={macro_f1:.4f}  weighted={weighted_f1:.4f}")

        if macro_f1 > best_macro:
            best_macro = macro_f1
            history["best_epoch"] = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_macro_f1": macro_f1}, checkpoint_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"    Early stopping at epoch {epoch}")
                break

    history["total_training_time_s"] = round(time.perf_counter() - t_total, 2)
    history["convergence_epoch"] = history["best_epoch"]
    history["avg_epoch_time_s"] = round(
        sum(history["epoch_times"]) / max(len(history["epoch_times"]), 1), 3
    )
    history["best_val_macro_f1"] = round(best_macro, 6)
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, Any]:
    model.eval()
    loader = make_loader(X, y, batch_size, shuffle=False)
    all_true, all_pred = [], []
    for Xb, yb in loader:
        preds = model(Xb.to(device)).argmax(1).cpu().numpy()
        all_true.extend(yb.numpy())
        all_pred.extend(preds)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    active  = sorted(np.unique(y_true).tolist())
    names   = [CLASS_NAMES[i] for i in active]
    report  = classification_report(
        y_true, y_pred, labels=active, target_names=names,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy":     round(accuracy_score(y_true, y_pred) * 100, 4),
        "macro_f1":     round(f1_score(y_true, y_pred, average="macro",    zero_division=0) * 100, 4),
        "weighted_f1":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 4),
        "per_class":    {n: {k: round(v, 4) for k, v in report[n].items()} for n in names},
        "n_samples":    len(y_true),
    }


def print_eval_table(model_type: str, test_name: str, metrics: Dict) -> None:
    print(f"\n  [{model_type.upper()}] on {test_name}")
    print(f"  Accuracy={metrics['accuracy']:.2f}%  "
          f"Macro-F1={metrics['macro_f1']:.2f}%  "
          f"Weighted-F1={metrics['weighted_f1']:.2f}%  "
          f"(n={metrics['n_samples']:,})")
    print(f"  {'Class':<14} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Sup':>8}")
    for cls, r in metrics["per_class"].items():
        print(f"  {cls:<14} {r.get('precision',0):>8.4f} "
              f"{r.get('recall',0):>8.4f} {r.get('f1-score',0):>8.4f} "
              f"{int(r.get('support',0)):>8}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # Load ablation config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ablation_name = cfg.get("ablation_name", "8class_comparison")
    out_dir  = Path(args.output) / "ciciot_binary_classification" / ablation_name / timestamp
    log_dir  = Path("logs") / ablation_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_dir), log_level="INFO", console_output=True, file_output=True)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")
    logger.info(f"Output: {out_dir}")

    data_root = Path(args.data_root)
    tr_cfg    = cfg.get("training_overrides", {})
    epochs    = tr_cfg.get("epochs", 80)
    patience  = tr_cfg.get("early_stopping_patience", 15)
    batch_sz  = tr_cfg.get("batch_size", 256)
    lr        = tr_cfg.get("learning_rate", 0.0005)
    oversample = tr_cfg.get("oversample_target", 5000)
    benign_cap = tr_cfg.get("benign_cap", 20000)
    zd_split  = cfg.get("zero_day_train_split", 0.8)
    s99_n     = cfg.get("test_99_1_sample_n", 100000)

    # ──────────────────────────────────────────────────────────────────────────
    # Load & merge training data
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading training data")
    logger.info("=" * 60)

    X_main, y_main = load_and_map(str(data_root / "ciciot_training_50_50.csv"))

    X_zd_all, y_zd_all = load_and_map(
        str(data_root / "zero_day_test.csv"), drop_benign=False
    )
    X_zd_tr, X_zd_test, y_zd_tr, y_zd_test = train_test_split(
        X_zd_all, y_zd_all, test_size=1 - zd_split, random_state=42, stratify=y_zd_all
    )
    logger.info(f"  zero_day → {len(X_zd_tr):,} train  |  {len(X_zd_test):,} test holdout (Mirai)")

    # Cap Benign to prevent it dominating training (raw file never modified)
    benign_idx = CLASS_TO_IDX["Benign"]
    benign_mask_tr = y_main == benign_idx
    n_benign = benign_mask_tr.sum()
    if n_benign > benign_cap:
        rng_cap = np.random.default_rng(42)
        keep_idx = rng_cap.choice(np.where(benign_mask_tr)[0], benign_cap, replace=False)
        attack_idx = np.where(~benign_mask_tr)[0]
        X_main = X_main[np.concatenate([keep_idx, attack_idx])]
        y_main = y_main[np.concatenate([keep_idx, attack_idx])]
        logger.info(f"  Benign capped from {n_benign:,} → {benign_cap:,} (in memory)")

    # Merge in memory
    X_train_raw = np.vstack([X_main, X_zd_tr])
    y_train_raw = np.concatenate([y_main, y_zd_tr])
    logger.info(f"  Merged training set: {len(X_train_raw):,} rows")

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_zd_test_sc   = scaler.transform(X_zd_test)

    # Oversample minority classes
    X_train_bal, y_train_bal = oversample_minority(X_train_scaled, y_train_raw, oversample)
    dist = {CLASS_NAMES[i]: int((y_train_bal == i).sum())
            for i in range(NUM_CLASSES) if (y_train_bal == i).sum() > 0}
    logger.info(f"  After oversampling (target={oversample}): {dist}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_bal, y_train_bal, test_size=0.15, random_state=42, stratify=y_train_bal
    )
    logger.info(f"  Train={len(X_tr):,}  Val={len(X_val):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # Load test sets
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Loading test sets ...")
    test_sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for fname in ["ciciot_test_5_95.csv", "ciciot_test_1_99.csv"]:
        X_t, y_t = load_and_map(str(data_root / fname))
        test_sets[fname.replace(".csv", "")] = (
            scaler.transform(X_t), y_t
        )

    X_t99, y_t99 = load_and_map(
        str(data_root / "ciciot_test_99_1.csv"), sample_n=s99_n
    )
    test_sets["ciciot_test_99_1_sampled"] = (scaler.transform(X_t99), y_t99)

    test_sets["zero_day_holdout_Mirai"] = (X_zd_test_sc, y_zd_test)

    for name, (X_t, y_t) in test_sets.items():
        dist_t = {CLASS_NAMES[i]: int((y_t == i).sum())
                  for i in range(NUM_CLASSES) if (y_t == i).sum() > 0}
        logger.info(f"  {name}: {len(X_t):,} rows → {dist_t}")

    # ──────────────────────────────────────────────────────────────────────────
    # DataLoaders
    # ──────────────────────────────────────────────────────────────────────────
    cw         = class_weights(y_tr, device)
    tr_loader  = make_loader(X_tr,  y_tr,  batch_sz, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_sz, shuffle=False)
    input_dim  = X_tr.shape[1]

    # Load base model config
    base_cfg = ConfigManager.load_config(
        str(Path(__file__).parent.parent / "configs" / "config.yaml"),
        profile=cfg.get("base_profile", "first"),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Run each model
    # ──────────────────────────────────────────────────────────────────────────
    experiment = cfg["experiments"][0]
    all_results: List[Dict] = []

    for model_spec in experiment["model_types"]:
        model_type = model_spec["type"]
        model_cfg  = {k: v for k, v in model_spec.items() if k != "type"}

        logger.info("\n" + "=" * 60)
        logger.info(f"  Model: {model_type.upper()}")
        logger.info("=" * 60)

        model = ModelFactory.create_model(
            input_seq_len=input_dim,
            num_classes=NUM_CLASSES,
            model_config={**vars(base_cfg.model), **model_cfg},
            model_type=model_type,
        ).to(device)

        n_params = model.count_parameters()
        logger.info(f"  Parameters: {n_params:,}")

        ckpt_path = str(out_dir / f"checkpoint_8class_{model_type}.pth")

        # Train
        hist = train_model(
            model=model, train_loader=tr_loader, val_loader=val_loader,
            device=device, epochs=epochs, lr=lr, patience=patience,
            checkpoint_path=ckpt_path, cw=cw,
        )

        # Load best checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"  Loaded best checkpoint (epoch {ckpt['epoch']}, "
                    f"macro_F1={ckpt['val_macro_f1']:.4f})")

        # Evaluate on all test sets
        test_results: Dict[str, Dict] = {}
        for test_name, (X_t, y_t) in test_sets.items():
            metrics = evaluate_on(model, X_t, y_t, device)
            test_results[test_name] = metrics
            print_eval_table(model_type, test_name, metrics)

        # Efficiency benchmark
        logger.info("  Running efficiency benchmark ...")
        try:
            bench = FullEfficiencyBenchmark(
                model=model, model_type=model_type,
                input_dim=input_dim, device=str(device),
                checkpoint_path=ckpt_path,
            )
            efficiency = bench.run_full_benchmark()
        except Exception as e:
            logger.warning(f"  Benchmark failed: {e}")
            efficiency = {}

        result = {
            "model_type":        model_type,
            "params":            n_params,
            "input_dim":         input_dim,
            "num_classes":       NUM_CLASSES,
            "training_history":  hist,
            "test_results":      test_results,
            "efficiency":        efficiency,
            "checkpoint_path":   ckpt_path,
        }
        all_results.append(result)

    # ──────────────────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────────────────
    summary = {
        "ablation_name": ablation_name,
        "timestamp":     timestamp,
        "device":        str(device),
        "classes":       CLASS_NAMES,
        "num_classes":   NUM_CLASSES,
        "training": {
            "sources": ["ciciot_training_50_50.csv", "zero_day_test.csv"],
            "train_rows":  len(X_tr),
            "val_rows":    len(X_val),
            "input_dim":   input_dim,
            "oversample_target": oversample,
        },
        "test_sets":     {k: {"n_samples": len(v[1])} for k, v in test_sets.items()},
        "experiments":   all_results,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # CSV — one row per model per test set
    rows = []
    for r in all_results:
        base = {
            "model":             r["model_type"],
            "params":            r["params"],
            "total_train_time_s": r["training_history"].get("total_training_time_s"),
            "best_epoch":        r["training_history"].get("best_epoch"),
            "convergence_epoch": r["training_history"].get("convergence_epoch"),
            "avg_epoch_time_s":  r["training_history"].get("avg_epoch_time_s"),
        }
        eff = r.get("efficiency", {})
        base["latency_p50_ms"]     = eff.get("latency_ms_p50")
        base["latency_p99_ms"]     = eff.get("latency_ms_p99")
        base["disk_mb"]            = eff.get("disk_mb")
        base["throughput"]         = eff.get("throughput_samples_per_s")
        base["complexity"]         = eff.get("complexity_class")
        base["edge_deployable"]    = eff.get("edge_deployable")

        for test_name, tm in r["test_results"].items():
            row = {**base, "test_set": test_name,
                   "accuracy": tm["accuracy"],
                   "macro_f1": tm["macro_f1"],
                   "weighted_f1": tm["weighted_f1"]}
            for cls_name, cr in tm.get("per_class", {}).items():
                row[f"{cls_name}_f1"] = round(cr.get("f1-score", 0) * 100, 4)
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    # Print summary table
    print("\n" + "=" * 78)
    print("  8-CLASS ABLATION — FINAL SUMMARY (7 attacks + Benign)")
    print("=" * 78)
    main_test = "ciciot_test_5_95"
    print(f"\n  Primary test set: {main_test}")
    print(f"  {'Model':<14} {'Acc':>8} {'MacroF1':>9} {'WtdF1':>9} "
          f"{'Params':>10} {'Lat_p50':>9} {'Train(s)':>10} {'Ep':>5}")
    print("  " + "-" * 76)
    for r in all_results:
        tm  = r["test_results"].get(main_test, {})
        eff = r.get("efficiency", {})
        hist = r["training_history"]
        print(f"  {r['model_type']:<14} "
              f"{tm.get('accuracy', 0):>8.2f}% "
              f"{tm.get('macro_f1', 0):>8.2f}% "
              f"{tm.get('weighted_f1', 0):>8.2f}% "
              f"{r['params']:>10,} "
              f"{eff.get('latency_ms_p50', float('nan')):>9.3f} "
              f"{hist.get('total_training_time_s', 0):>10.1f}s "
              f"{hist.get('best_epoch', 0):>5}")

    print(f"\n  Mirai test (zero_day holdout):")
    print(f"  {'Model':<14} {'Acc':>8} {'MacroF1':>9} {'Mirai_F1':>10}")
    print("  " + "-" * 45)
    for r in all_results:
        tm = r["test_results"].get("zero_day_holdout_Mirai", {})
        mf1 = tm.get("per_class", {}).get("Mirai", {}).get("f1-score", 0) * 100
        print(f"  {r['model_type']:<14} "
              f"{tm.get('accuracy', 0):>8.2f}% "
              f"{tm.get('macro_f1', 0):>8.2f}% "
              f"{mf1:>10.2f}%")

    print(f"\n  Results saved to: {out_dir}/")
    print(f"  Files: results.json, results.csv, checkpoint_7class_*.pth\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="8-class classification ablation (7 attacks + Benign)")
    parser.add_argument("--config",    required=True,
                        help="Path to 8class_comparison.yaml")
    parser.add_argument("--data-root", default="data/raw/ciciot_2023",
                        help="Directory containing all CIC-IoT CSV files")
    parser.add_argument("--output",    default="results",
                        help="Base output directory")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
