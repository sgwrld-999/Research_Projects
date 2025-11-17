import os
from pathlib import Path
import logging
from datetime import datetime
import json
import numpy as np
import random
import torch


def set_global_seed(seed: int = 42):
    """Set random seeds for reproducibility across PyTorch, NumPy, and Python."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    """Ensure directory exists, accepts str or Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_id():
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def get_logger(name: str, run_id: str = None, log_dir = None, level=logging.INFO):
    if run_id is None:
        run_id = get_run_id()
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / 'logs'
    else:
        log_dir = Path(log_dir)
    log_dir = log_dir.resolve()
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # file handler
        logfile = log_dir / f"{run_id}.log"
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj, path):
    """Save JSON to path, accepts str or Path."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    """Load JSON from path, accepts str or Path."""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)
