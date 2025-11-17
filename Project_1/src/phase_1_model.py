from pathlib import Path
import json
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from utils import get_logger, ensure_dir, save_json

LOGGER = get_logger('phase1_model')

# Set device for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER.info(f'Using device: {device}')
if torch.cuda.is_available():
    LOGGER.info(f'CUDA Device: {torch.cuda.get_device_name(0)}')
    LOGGER.info(f'CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')


class Encoder(nn.Module):
    """VAE Encoder Network"""
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        
        if input_dim > 100:
            sizes = [256, 128, 64]
        elif input_dim > 50:
            sizes = [128, 64, 32]
        else:
            sizes = [64, 32, 16]
        
        layers = []
        prev_size = input_dim
        for size in sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout)
            ])
            prev_size = size
        
        self.encoder_layers = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(prev_size, latent_dim)
        self.fc_log_var = nn.Linear(prev_size, latent_dim)
    
    def forward(self, x):
        h = self.encoder_layers(x)
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var


class Decoder(nn.Module):
    """VAE Decoder Network"""
    def __init__(self, latent_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        
        if output_dim > 100:
            sizes = [64, 128, 256]
        elif output_dim > 50:
            sizes = [32, 64, 128]
        else:
            sizes = [16, 32, 64]
        
        layers = []
        prev_size = latent_dim
        for size in sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout)
            ])
            prev_size = size
        
        layers.append(nn.Linear(prev_size, output_dim))
        layers.append(nn.Sigmoid())
        
        self.decoder_layers = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder_layers(z)


class Layer1AutoencoderVAE(nn.Module):
    """Variational Autoencoder for Phase-1 Anomaly Detection"""
    def __init__(self, input_dim: int, latent_dim: int = None, l2_reg: float = 1e-5, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        if latent_dim is None:
            self.latent_dim = min(12, max(3, input_dim // 2))
        else:
            self.latent_dim = latent_dim
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.beta = 1.0
        
        self.encoder = Encoder(input_dim, self.latent_dim, dropout)
        self.decoder = Decoder(self.latent_dim, input_dim, dropout)
        
        # Move model to device
        self.to(device)
        LOGGER.info(f'VAE initialized on {device}')
    
    def reparameterize(self, z_mean, z_log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        recon = self.decoder(z)
        return recon, z_mean, z_log_var
    
    def loss_function(self, recon_x, x, z_mean, z_log_var):
        """Compute VAE loss (Reconstruction + KL Divergence)"""
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def train_phase1(X_train: np.ndarray, X_val: Optional[np.ndarray] = None, input_dim: Optional[int] = None, 
                epochs: int = 30, batch_size: int = 64, run_dir: str = 'models', logger=None) -> Tuple[Layer1AutoencoderVAE, dict]:
    """
    Train Phase-1 VAE model using PyTorch with CUDA support.
    
    Args:
        X_train: Training data (numpy array)
        X_val: Validation data (optional)
        input_dim: Input dimensionality
        epochs: Number of training epochs
        batch_size: Batch size for training
        run_dir: Directory to save models
        logger: Logger instance
    
    Returns:
        Tuple of (trained VAE model, training history)
    """
    if logger is None:
        logger = LOGGER
    
    input_dim = int(X_train.shape[1]) if input_dim is None else input_dim
    
    # Create VAE model
    vae = Layer1AutoencoderVAE(input_dim=input_dim)
    
    # Setup optimizer with L2 regularization (weight decay)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4, weight_decay=vae.l2_reg)
    
    ensure_dir(run_dir)
    logger.info(f'Starting Phase-1 training with input_dim={input_dim}, latent_dim={vae.latent_dim}')
    logger.info(f'Using device: {device}')
    
    # Create data loaders
    train_tensor = torch.FloatTensor(X_train).to(device)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None and len(X_val) > 0:
        val_tensor = torch.FloatTensor(X_val).to(device)
        val_dataset = TensorDataset(val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training history
    history = {'loss': [], 'recon_loss': [], 'kl_loss': [], 'val_loss': []}
    
    # Training loop
    vae.train()
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        epoch_recon_losses = []
        epoch_kl_losses = []
        
        for batch_idx, (batch_data,) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, z_mean, z_log_var = vae(batch_data)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = vae.loss_function(recon_batch, batch_data, z_mean, z_log_var)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_losses.append(total_loss.item())
            epoch_recon_losses.append(recon_loss.item())
            epoch_kl_losses.append(kl_loss.item())
        
        # Calculate epoch averages
        mean_loss = float(np.mean(epoch_losses))
        mean_recon_loss = float(np.mean(epoch_recon_losses))
        mean_kl_loss = float(np.mean(epoch_kl_losses))
        
        history['loss'].append(mean_loss)
        history['recon_loss'].append(mean_recon_loss)
        history['kl_loss'].append(mean_kl_loss)
        
        # Validation
        if val_loader is not None:
            vae.eval()
            val_losses = []
            with torch.no_grad():
                for (val_batch,) in val_loader:
                    recon_val, z_mean_val, z_log_var_val = vae(val_batch)
                    val_loss, _, _ = vae.loss_function(recon_val, val_batch, z_mean_val, z_log_var_val)
                    val_losses.append(val_loss.item())
            mean_val_loss = float(np.mean(val_losses))
            history['val_loss'].append(mean_val_loss)
            vae.train()
            logger.info(f'Epoch {epoch}/{epochs} - loss: {mean_loss:.6f} - val_loss: {mean_val_loss:.6f}')
        else:
            logger.info(f'Epoch {epoch}/{epochs} - loss: {mean_loss:.6f}')
    
    # Save model
    run_dir = Path(run_dir)
    try:
        encoder_path = run_dir / 'encoder.pth'
        decoder_path = run_dir / 'decoder.pth'
        full_model_path = run_dir / 'vae_full.pth'
        
        torch.save(vae.encoder.state_dict(), encoder_path)
        torch.save(vae.decoder.state_dict(), decoder_path)
        torch.save(vae.state_dict(), full_model_path)
        
        logger.info(f'Saved encoder to {encoder_path}')
        logger.info(f'Saved decoder to {decoder_path}')
        logger.info(f'Saved full model to {full_model_path}')
    except Exception as e:
        logger.warning(f'Failed to save models: {e}')
    
    return vae, history


def compute_reconstruction_losses(vae: Layer1AutoencoderVAE, X: np.ndarray) -> np.ndarray:
    """
    Compute reconstruction losses for anomaly detection.
    
    Args:
        vae: Trained VAE model
        X: Input data (numpy array)
    
    Returns:
        Array of reconstruction losses (MSE per sample)
    """
    vae.eval()
    
    # Convert to tensor and move to device
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        recon, _, _ = vae(X_tensor)
        # Compute MSE per sample
        mse = torch.mean((X_tensor - recon) ** 2, dim=1)
        mse = mse.cpu().numpy()
    
    LOGGER.info(f'Computed reconstruction losses for {len(mse)} samples')
    return mse


def compute_kde_threshold(losses: np.ndarray, bandwidth: float = 0.1, percentile: float = 95.0, logger=None) -> dict:
    if logger is None:
        logger = LOGGER
    losses = losses.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(losses)
    # sample density and pick threshold as percentile of losses
    thresh = float(np.percentile(losses, percentile))
    logger.info(f'KDE threshold at {percentile} percentile is {thresh:.6f} (bandwidth={bandwidth})')
    return {'threshold': thresh, 'bandwidth': bandwidth}


def save_threshold(cfg: dict, path):
    """Save threshold config to JSON. Accepts str or Path."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    LOGGER.info(f'Saved threshold cfg to {path}')
