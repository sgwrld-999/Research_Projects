import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from typing import Dict, Any, Optional
from utils import get_logger

LOGGER = get_logger('metrics')


def confusion_counts(y_true, y_pred) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)}


def phase1_metrics(y_true, losses, threshold: float, logger=None) -> Dict[str, Any]:
    if logger is None:
        logger = LOGGER
    y_pred = (losses > threshold).astype(int)
    y_true_bin = np.array([1 if v != 0 and v != 'Normal' and v != 'normal' else 0 for v in y_true])
    counts = confusion_counts(y_true_bin, y_pred)
    tpr = recall_score(y_true_bin, y_pred)
    fpr = counts['FP'] / (counts['FP'] + counts['TN']) if (counts['FP'] + counts['TN']) > 0 else 0.0
    auc = None
    try:
        auc = roc_auc_score(y_true_bin, losses)
    except Exception:
        auc = None
    res = {'TPR': float(tpr), 'FPR': float(fpr), 'AUC': auc, 'confusion': counts}
    logger.info(f'Phase-1 metrics: {res}')
    return res


def phase2_metrics(y_true, y_pred, labels=None, logger=None) -> Dict[str, Any]:
    if logger is None:
        logger = LOGGER
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    counts = confusion_matrix(y_true, y_pred)
    res = {'accuracy': float(acc), 'precision_weighted': float(prec), 'recall_weighted': float(rec), 'f1_weighted': float(f1), 'confusion_matrix': counts.tolist()}
    logger.info(f'Phase-2 metrics: {res}')
    return res
