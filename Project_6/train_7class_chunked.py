"""7-Class Attack Classifier — Chunked Training on CIC-IoT 2023.

Classes:
    0  DDoS       — ACK/UDP/TCP/ICMP/SYN/PSHACK/RSTFIN/HTTP/SlowLoris/SynonymousIP/Frag floods
    1  DoS        — TCP/UDP/SYN/HTTP flood (single-source, lower-rate variant)
    2  Spoofing   — ARP spoofing, DNS spoofing
    3  Recon      — Ping sweep, OS/Port/Vuln scan, Host discovery
    4  Web-based  — SQLi, Command injection, Backdoor, Upload attack, XSS, Browser hijack
    5  Brute Force — Dictionary brute force
    6  Mirai      — GREIP flood, Greeth flood, UDPPlain (zero-day data only)

Training strategy (chunked — raw files never merged or modified):
    Phase 1  ciciot_training_50_50.csv   → learns classes 0-5 (Mirai slot exists but empty)
    Phase 2  zero_day_test.csv           → fine-tunes class 6 (Mirai) via experience replay
    Test     ciciot_test_5_95.csv        → evaluates classes 0-5
             zero_day 20% holdout        → evaluates class 6

Preprocessing:
    - Only the 37 features common to ALL three files are used
    - StandardScaler fitted on Phase 1 data only; reused for Phase 2 and test (no leakage)
    - Raw files are read-only; nothing is written back

Usage:
    python train_7class_chunked.py
    python train_7class_chunked.py --device cuda --epochs1 60 --epochs2 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "lightweight-transformer-ids"))
from src.model import LinformerIDS
from src.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# 37 features common across ciciot_training_50_50, zero_day_test, ciciot_test_*
COMMON_FEATURES = [
    "ARP", "AVG", "DHCP", "DNS", "HTTP", "HTTPS", "Header_Length", "IAT",
    "ICMP", "IPv", "IRC", "LLC", "Max", "Min", "Number", "Protocol Type",
    "Rate", "SMTP", "SSH", "Std", "TCP", "Telnet", "Tot size", "Tot sum",
    "UDP", "Variance", "ack_count", "ack_flag_number", "cwr_flag_number",
    "ece_flag_number", "fin_count", "fin_flag_number", "psh_flag_number",
    "rst_count", "rst_flag_number", "syn_count", "syn_flag_number",
]

CLASS_NAMES = ["DDoS", "DoS", "Spoofing", "Recon", "Web-based", "Brute Force", "Mirai"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES   = len(CLASS_NAMES)

# Raw label → super-class mapping (ciciot + zero_day labels)
LABEL_MAP: Dict[str, str] = {
    # ── DDoS ──────────────────────────────────────────────────────────────────
    "DDOS-ACK_FRAGMENTATION":      "DDoS",
    "DDOS-UDP_FLOOD":              "DDoS",
    "DDOS-SLOWLORIS":              "DDoS",
    "DDOS-ICMP_FLOOD":             "DDoS",
    "DDOS-RSTFINFLOOD":            "DDoS",
    "DDOS-PSHACK_FLOOD":           "DDoS",
    "DDOS-HTTP_FLOOD":             "DDoS",
    "DDOS-UDP_FRAGMENTATION":      "DDoS",
    "DDOS-TCP_FLOOD":              "DDoS",
    "DDOS-SYN_FLOOD":              "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD":     "DDoS",
    "DDOS-ICMP_FRAGMENTATION":     "DDoS",
    # ── DoS ───────────────────────────────────────────────────────────────────
    "DOS-TCP_FLOOD":               "DoS",
    "DOS-HTTP_FLOOD":              "DoS",
    "DOS-SYN_FLOOD":               "DoS",
    "DOS-UDP_FLOOD":               "DoS",
    # ── Spoofing ──────────────────────────────────────────────────────────────
    "MITM-ARPSPOOFING":            "Spoofing",
    "DNS_SPOOFING":                "Spoofing",
    # ── Recon ─────────────────────────────────────────────────────────────────
    "RECONNAISSANCE-PINGSWEEP":    "Recon",
    "RECONNAISSANCE-OSSCAN":       "Recon",
    "VULNERABILITYSCAN":           "Recon",
    "RECONNAISSANCE-PORTSCAN":     "Recon",
    "RECONNAISSANCE-HOSTDISCOVERY":"Recon",
    "RECON-HOSTDISCOVERY":         "Recon",
    "RECON-OSSCAN":                "Recon",
    "RECON-PORTSCAN":              "Recon",
    "RECON-PINGSWEEP":             "Recon",
    # ── Web-based ─────────────────────────────────────────────────────────────
    "SQLINJECTION":                "Web-based",
    "COMMANDINJECTION":            "Web-based",
    "BACKDOOR_MALWARE":            "Web-based",
    "UPLOADING_ATTACK":            "Web-based",
    "XSS":                         "Web-based",
    "BROWSERHIJACKING":            "Web-based",
    # ── Brute Force ───────────────────────────────────────────────────────────
    "DICTIONARYBRUTEFORCE":        "Brute Force",
    # ── Mirai (zero-day only) ─────────────────────────────────────────────────
    "MIRAI-GREIP_FLOOD":           "Mirai",
    "MIRAI-GREETH_FLOOD":          "Mirai",
    "MIRAI-UDPPLAIN":              "Mirai",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading — read-only, no file writes
# ──────────────────────────────────────────────────────────────────────────────

def _label_col(df: pd.DataFrame) -> str:
    """Return the label column name case-insensitively."""
    for c in df.columns:
        if c.lower() == "label":
            return c
    raise ValueError(f"No 'label' column found. Columns: {df.columns.tolist()}")


def load_and_map(
    csv_path: str,
    drop_benign: bool = True,
    drop_unmapped: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load CSV, extract common features and map labels to 7 super-classes.

    Args:
        csv_path: Path to raw CSV (never modified).
        drop_benign: Drop BENIGN/BenignTraffic rows (default True — 7 attack classes only).
        drop_unmapped: Drop rows whose label isn't in LABEL_MAP.

    Returns:
        X (n, 37), y (n,) int64, original_labels list
    """
    logger.info(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    lc = _label_col(df)

    raw_labels = df[lc].astype(str).str.strip().str.upper()
    df[lc] = df[lc].astype(str).str.strip()

    if drop_benign:
        benign_mask = raw_labels.isin({"BENIGN", "BENIGNTRAFFIC", "NORMAL"})
        df = df[~benign_mask].reset_index(drop=True)
        raw_labels = df[lc].astype(str).str.strip()
        logger.info(f"  After dropping BENIGN: {len(df)} rows")

    # Map labels — use original case for dict lookup
    mapped = df[lc].map(LABEL_MAP)
    if drop_unmapped:
        mask = mapped.notna()
        n_dropped = (~mask).sum()
        if n_dropped:
            unmapped = df[lc][~mask].unique()
            logger.warning(f"  Dropping {n_dropped} rows with unmapped labels: {unmapped[:10]}")
        df = df[mask].reset_index(drop=True)
        mapped = mapped[mask].reset_index(drop=True)

    # Extract only the 37 common features
    available = [c for c in COMMON_FEATURES if c in df.columns]
    missing   = [c for c in COMMON_FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"  Missing common features (will be zero-filled): {missing}")

    X_df = df[available].copy()
    for m in missing:
        X_df[m] = 0.0
    X_df = X_df[COMMON_FEATURES]  # enforce column order

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    X = X_df.values.astype(np.float32)
    y = np.array([CLASS_TO_IDX[c] for c in mapped], dtype=np.int64)

    dist = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(NUM_CLASSES) if (y == i).sum() > 0}
    logger.info(f"  Loaded: X={X.shape}  Class distribution: {dist}")
    return X, y, df[lc].tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def fit_scaler(X: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on Phase 1 data. Returns fitted scaler."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ──────────────────────────────────────────────────────────────────────────────

def _oversample(
    X: np.ndarray,
    y: np.ndarray,
    target: int,
    num_classes: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random oversampling of minority classes to `target` samples each.

    Majority classes above `target` are left untouched (no undersampling).
    Purely in-memory — raw files are never modified.
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [X], [y]
    for cls in range(num_classes):
        mask = y == cls
        n = mask.sum()
        if 0 < n < target:
            need = target - n
            idx  = rng.choice(np.where(mask)[0], need, replace=True)
            X_parts.append(X[idx])
            y_parts.append(y[idx])
    return np.vstack(X_parts), np.concatenate(y_parts)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=shuffle)


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score: Optional[float] = None

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to handle severe imbalance."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)          # avoid /0 for missing classes
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes    # normalise so mean weight = 1
    return torch.FloatTensor(weights)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip: float = 1.0,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        n += len(y_b)
    return total_loss / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for X_b, y_b in loader:
        preds = model(X_b.to(device)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y_b.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def run_phase(
    phase: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: str,
    early_stopping_patience: int = 10,
    f1_average: str = "weighted",
) -> Dict:
    """Generic training loop for a single phase. Returns training history.

    Args:
        f1_average: F1 averaging for checkpoint selection and early stopping.
            Phase 1 → 'weighted': avoids penalising on absent/tiny test classes.
            Phase 2 → 'macro': ensures new Mirai class is explicitly rewarded.
    """
    stopper  = EarlyStopping(patience=early_stopping_patience)
    best_f1  = 0.0
    history  = {"train_loss": [], "val_macro_f1": [], "val_weighted_f1": [], "best_epoch": 0}
    t_start  = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        monitor_f1  = weighted_f1 if f1_average == "weighted" else macro_f1

        history["train_loss"].append(round(loss, 6))
        history["val_macro_f1"].append(round(macro_f1, 6))
        history["val_weighted_f1"].append(round(weighted_f1, 6))

        if monitor_f1 > best_f1:
            best_f1 = monitor_f1
            history["best_epoch"] = epoch
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_f1": monitor_f1,
                        "val_macro_f1": macro_f1,
                        "val_weighted_f1": weighted_f1}, checkpoint_path)

        print(f"  [Phase {phase}] Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  "
              f"macro={macro_f1:.4f}  weighted={weighted_f1:.4f}  "
              f"best({f1_average})={best_f1:.4f}")

        if stopper(monitor_f1):
            print(f"  Early stopping at epoch {epoch}")
            break

    history["total_time_s"] = round(time.perf_counter() - t_start, 2)
    history["best_val_f1"]  = round(best_f1, 6)
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Results printing & saving
# ──────────────────────────────────────────────────────────────────────────────

def print_results(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    active_classes: Optional[List[int]] = None,
) -> Dict:
    """Print per-class metrics table and return metrics dict."""
    labels = active_classes if active_classes else list(range(NUM_CLASSES))
    names  = [CLASS_NAMES[i] for i in labels]

    acc     = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
    w_f1    = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  Accuracy : {acc*100:.4f}%")
    print(f"  Macro F1 : {macro_f1*100:.4f}%")
    print(f"  Wtd   F1 : {w_f1*100:.4f}%")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*54}")

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names,
        output_dict=True, zero_division=0,
    )
    per_class = {}
    for name in names:
        r = report.get(name, {})
        print(f"  {name:<14} {r.get('precision',0):>10.4f} "
              f"{r.get('recall',0):>10.4f} {r.get('f1-score',0):>10.4f} "
              f"{int(r.get('support',0)):>10}")
        per_class[name] = r
    print(f"{'='*70}\n")

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": w_f1,
            "per_class": per_class}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir   = Path(args.output) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")
    print(f"Output: {out_dir}\n")

    data_root = Path(args.data_root)
    ckpt_p1   = str(out_dir / "checkpoint_phase1.pth")
    ckpt_p2   = str(out_dir / "checkpoint_phase2.pth")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Train on ciciot_training_50_50.csv (classes 0-5, no Mirai)
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  PHASE 1 — Training on ciciot_training_50_50.csv (6 classes)")
    print("=" * 70)

    X_p1, y_p1, _ = load_and_map(str(data_root / "ciciot_training_50_50.csv"))

    # Fit scaler ONLY on Phase 1 data
    scaler = fit_scaler(X_p1)
    X_p1   = scaler.transform(X_p1)

    X_tr1, X_val1, y_tr1, y_val1 = train_test_split(
        X_p1, y_p1, test_size=0.2, random_state=42, stratify=y_p1
    )

    # Build experience replay buffer BEFORE scaling further (X_p1 already scaled)
    # 500 samples per active class → stored in memory for Phase 2
    replay_X: List[np.ndarray] = []
    replay_y: List[np.ndarray] = []
    for cls_idx in range(6):                             # classes 0-5 only
        mask = y_p1 == cls_idx
        if mask.sum() == 0:
            continue
        n_replay = min(500, mask.sum())
        idx = np.random.default_rng(42).choice(np.where(mask)[0], n_replay, replace=False)
        replay_X.append(X_p1[idx])
        replay_y.append(y_p1[idx])
    replay_X_arr = np.vstack(replay_X)
    replay_y_arr = np.concatenate(replay_y)
    print(f"\nExperience replay buffer: {len(replay_y_arr)} samples "
          f"({dict(zip(*np.unique(replay_y_arr, return_counts=True)))})\n")

    # Class weights for imbalanced Phase 1 data
    cw_p1 = compute_class_weights(y_tr1, NUM_CLASSES).to(device)

    val1_loader = make_loader(X_val1, y_val1, args.batch_size, shuffle=False)

    model = LinformerIDS(
        input_seq_len=len(COMMON_FEATURES),
        num_classes=NUM_CLASSES,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        k=args.k,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Oversample minority classes in memory to balance training.
    # Purely in-memory — raw files never modified.
    target_per_class = args.oversample_target
    X_tr1_bal, y_tr1_bal = _oversample(X_tr1, y_tr1, target_per_class, NUM_CLASSES)
    dist_bal = {CLASS_NAMES[i]: int((y_tr1_bal == i).sum())
                for i in range(NUM_CLASSES) if (y_tr1_bal == i).sum() > 0}
    print(f"\n  Phase 1 training (after oversampling to ~{target_per_class}/class): {dist_bal}")

    tr1_loader = make_loader(X_tr1_bal, y_tr1_bal, args.batch_size, shuffle=True)

    cw_p1 = compute_class_weights(y_tr1_bal, NUM_CLASSES).to(device)
    criterion_p1 = nn.CrossEntropyLoss(weight=cw_p1)
    optimizer_p1 = optim.Adam(model.parameters(), lr=args.lr1, weight_decay=1e-4)
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=args.epochs1)

    print(f"\nPhase 1: {args.epochs1} epochs, lr={args.lr1}, batch={args.batch_size}, "
          f"class-weighted CE + oversampling, checkpoint on weighted-F1\n")

    hist_p1 = run_phase(
        phase=1, model=model,
        train_loader=tr1_loader, val_loader=val1_loader,
        criterion=criterion_p1, optimizer=optimizer_p1,
        device=device, epochs=args.epochs1,
        checkpoint_path=ckpt_p1,
        early_stopping_patience=args.patience1,
        f1_average="weighted",       # weighted F1 — not penalised by absent Brute Force/Web
    )

    # Advance scheduler for each epoch trained
    for _ in range(len(hist_p1["train_loss"])):
        scheduler_p1.step()

    # Load best Phase 1 weights before Phase 2
    ckpt = torch.load(ckpt_p1, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded Phase 1 best checkpoint (epoch {ckpt['epoch']}, "
          f"val_F1={ckpt['val_f1']:.4f})\n")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tune on zero_day_test.csv (Mirai, class 6)
    #           with experience replay from Phase 1
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  PHASE 2 — Fine-tuning on zero_day_test.csv (Mirai, class 6)")
    print("=" * 70)

    X_zd, y_zd, _ = load_and_map(str(data_root / "zero_day_test.csv"), drop_benign=False)
    X_zd = scaler.transform(X_zd)          # Phase 1 scaler — no refitting

    # 80% fine-tuning, 20% held-out test for Mirai evaluation
    X_zd_tr, X_zd_test, y_zd_tr, y_zd_test = train_test_split(
        X_zd, y_zd, test_size=0.2, random_state=42, stratify=y_zd
    )
    print(f"\n  zero_day fine-tune: {len(X_zd_tr):,}  |  Mirai test holdout: {len(X_zd_test):,}")

    # Cap Mirai fine-tune samples to prevent drowning the replay buffer.
    # Ratio target: Mirai samples ≈ 5× replay buffer so experience replay stays meaningful.
    max_mirai_finetune = min(len(X_zd_tr), args.max_mirai)
    rng_zd = np.random.default_rng(42)
    zd_idx = rng_zd.choice(len(X_zd_tr), max_mirai_finetune, replace=False)
    X_zd_tr_cap = X_zd_tr[zd_idx]
    y_zd_tr_cap = y_zd_tr[zd_idx]
    print(f"  Mirai fine-tune capped at {max_mirai_finetune:,} samples "
          f"(replay ratio ≈ 1:{max_mirai_finetune // max(len(replay_y_arr), 1)})")

    # Mix: replay buffer + capped Mirai fine-tune data
    X_p2 = np.vstack([replay_X_arr, X_zd_tr_cap])
    y_p2 = np.concatenate([replay_y_arr, y_zd_tr_cap])

    dist_p2 = {CLASS_NAMES[i]: int((y_p2 == i).sum())
               for i in range(NUM_CLASSES) if (y_p2 == i).sum() > 0}
    print(f"  Phase 2 training set: {len(X_p2):,} samples — {dist_p2}\n")

    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_p2, y_p2, test_size=0.1, random_state=42, stratify=y_p2
    )

    cw_p2 = compute_class_weights(y_tr2, NUM_CLASSES).to(device)

    tr2_loader  = make_loader(X_tr2,  y_tr2,  args.batch_size, shuffle=True)
    val2_loader = make_loader(X_val2, y_val2, args.batch_size, shuffle=False)

    # Lower LR to avoid catastrophic forgetting of Phase 1 knowledge
    optimizer_p2 = optim.Adam(model.parameters(), lr=args.lr2, weight_decay=1e-4)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=args.epochs2)
    criterion_p2 = nn.CrossEntropyLoss(weight=cw_p2)

    print(f"Phase 2: {args.epochs2} epochs, lr={args.lr2} (10× lower), "
          f"batch={args.batch_size}, experience replay, class-weighted CE\n")

    hist_p2 = run_phase(
        phase=2, model=model,
        train_loader=tr2_loader, val_loader=val2_loader,
        criterion=criterion_p2, optimizer=optimizer_p2,
        device=device, epochs=args.epochs2,
        checkpoint_path=ckpt_p2,
        early_stopping_patience=args.patience2,
        f1_average="macro",          # macro F1 — ensures Mirai (new class) is explicitly rewarded
    )

    for _ in range(len(hist_p2["train_loss"])):
        scheduler_p2.step()

    # Load best Phase 2 weights for evaluation
    ckpt = torch.load(ckpt_p2, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nLoaded Phase 2 best checkpoint (epoch {ckpt['epoch']}, "
          f"val_F1={ckpt['val_f1']:.4f})\n")

    # ──────────────────────────────────────────────────────────────────────────
    # TESTING — Separate evaluation on each dataset (no merging)
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  TESTING")
    print("=" * 70)

    all_results: Dict = {}

    # Test A: ciciot_test_5_95.csv (attack rows → classes 0-5)
    print("\n[Test A] ciciot_test_5_95.csv — classes 0-5 (DDoS, DoS, Spoofing, Recon, Web, BF)")
    X_tA, y_tA, _ = load_and_map(str(data_root / "ciciot_test_5_95.csv"))
    X_tA = scaler.transform(X_tA)
    loader_tA = make_loader(X_tA, y_tA, args.batch_size, shuffle=False)
    y_true_A, y_pred_A = evaluate(model, loader_tA, device)
    active_A = sorted(np.unique(y_true_A).tolist())
    res_A = print_results("Test A — ciciot_test_5_95 (6 attack classes)", y_true_A, y_pred_A, active_A)
    all_results["test_A_ciciot"] = res_A

    # Test B: zero_day 20% holdout (Mirai, class 6)
    print("[Test B] zero_day holdout — class 6 (Mirai)")
    loader_tB = make_loader(X_zd_test, y_zd_test, args.batch_size, shuffle=False)
    y_true_B, y_pred_B = evaluate(model, loader_tB, device)
    active_B = sorted(np.unique(y_true_B).tolist())
    res_B = print_results("Test B — zero_day holdout (Mirai class)", y_true_B, y_pred_B, active_B)
    all_results["test_B_mirai"] = res_B

    # Combined 7-class summary (computed from concatenated separate test results)
    print("[Combined] 7-class summary across both test sets")
    y_true_all = np.concatenate([y_true_A, y_true_B])
    y_pred_all = np.concatenate([y_pred_A, y_pred_B])
    res_comb = print_results("Combined — All 7 Attack Classes", y_true_all, y_pred_all)
    all_results["combined_7class"] = res_comb

    # ──────────────────────────────────────────────────────────────────────────
    # Save everything
    # ──────────────────────────────────────────────────────────────────────────
    final = {
        "timestamp": timestamp,
        "args": vars(args),
        "classes": CLASS_NAMES,
        "phase1_training": hist_p1,
        "phase2_finetuning": hist_p2,
        "results": all_results,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    # CSV summary
    rows = []
    for split_name, res in all_results.items():
        row = {"split": split_name,
               "accuracy": round(res["accuracy"] * 100, 4),
               "macro_f1": round(res["macro_f1"] * 100, 4),
               "weighted_f1": round(res["weighted_f1"] * 100, 4)}
        for cls_name, cr in res.get("per_class", {}).items():
            row[f"{cls_name}_f1"] = round(cr.get("f1-score", 0) * 100, 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    print(f"\nAll results saved to: {out_dir}/")
    print(f"  results.json, results.csv")
    print(f"  checkpoint_phase1.pth, checkpoint_phase2.pth\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7-class chunked training for CIC-IoT")

    # Paths
    parser.add_argument("--data-root", default="lightweight-transformer-ids/data/raw/ciciot_2023",
                        help="Directory containing all CIC-IoT CSV files")
    parser.add_argument("--output",    default="results/7class_chunked",
                        help="Output directory for checkpoints and results")

    # Device
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")

    # Model architecture
    parser.add_argument("--dim",      type=int,   default=64)
    parser.add_argument("--depth",    type=int,   default=4)
    parser.add_argument("--heads",    type=int,   default=4)
    parser.add_argument("--k",        type=int,   default=16)
    parser.add_argument("--dropout",  type=float, default=0.1)

    # Training
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--epochs1",    type=int,   default=50,   help="Phase 1 epochs")
    parser.add_argument("--epochs2",    type=int,   default=20,   help="Phase 2 epochs")
    parser.add_argument("--lr1",        type=float, default=0.001, help="Phase 1 learning rate")
    parser.add_argument("--lr2",        type=float, default=0.0001,help="Phase 2 learning rate (lower)")
    parser.add_argument("--patience1",  type=int,   default=10,   help="Phase 1 early stop patience")
    parser.add_argument("--patience2",  type=int,   default=7,    help="Phase 2 early stop patience")
    parser.add_argument("--oversample-target", type=int, default=1000,
                        help="Oversample minority Phase 1 classes to this many samples (in memory)")
    parser.add_argument("--max-mirai",  type=int,   default=10000,
                        help="Max Mirai samples for Phase 2 fine-tuning (prevents replay drowning)")

    args = parser.parse_args()
    main(args)
