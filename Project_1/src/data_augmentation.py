import numpy as np
from typing import List, Tuple
from utils import get_logger

LOGGER = get_logger('data_augmentation')

def jitter(x: np.ndarray, sigma=0.03) -> np.ndarray:
    """
    Apply Jittering (adding Gaussian noise) to the time series.
    https://arxiv.org/pdf/1706.00527.pdf
    """
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x: np.ndarray, sigma=0.1) -> np.ndarray:
    """
    Apply Scaling (multiplying by random scalar) to the time series.
    https://arxiv.org/pdf/1706.00527.pdf
    """
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def augment_minority_classes(X: np.ndarray, y: np.ndarray, target_classes: List[int], multiplier: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment specified minority classes using Jittering and Scaling.
    
    Args:
        X: Input features (N, seq_len, features)
        y: Labels (N,)
        target_classes: List of class indices to augment
        multiplier: How many times to replicate each sample (e.g., 1 means double the size)
        
    Returns:
        X_aug, y_aug: Augmented dataset including original samples
    """
    LOGGER.info(f"Starting augmentation for classes {target_classes} with multiplier {multiplier}")
    
    X_new = [X]
    y_new = [y]
    
    for cls in target_classes:
        # Find indices of the class
        indices = np.where(y == cls)[0]
        n_samples = len(indices)
        
        if n_samples == 0:
            LOGGER.warning(f"No samples found for class {cls}. Skipping.")
            continue
            
        LOGGER.info(f"Augmenting class {cls}: {n_samples} samples")
        
        X_cls = X[indices]
        y_cls = y[indices]
        
        for _ in range(multiplier):
            # Apply Jittering
            X_jittered = jitter(X_cls)
            X_new.append(X_jittered)
            y_new.append(y_cls)
            
            # Apply Scaling
            X_scaled = scaling(X_cls)
            X_new.append(X_scaled)
            y_new.append(y_cls)
            
    X_aug = np.concatenate(X_new, axis=0)
    y_aug = np.concatenate(y_new, axis=0)
    
    LOGGER.info(f"Augmentation complete. Original shape: {X.shape}, New shape: {X_aug.shape}")
    return X_aug, y_aug
