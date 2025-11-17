import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from multi_class_data_processing import preprocess_multiclass_data, drop_zero_value_features, drop_zero_variance_columns, handle_missing_values, engineer_time_features, apply_onehot_encoding, apply_label_encoding, apply_hash_encoding, convert_object_to_numeric, apply_minmax_scaling, apply_log_transformation, apply_zscore_normalization, create_aggregated_features
from utils import get_logger, ensure_dir, set_global_seed, save_json
from plots import plot_confusion_matrix

LOGGER = get_logger('phase1_vae')

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super(VAE, self).__init__()
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.Dropout(0.2))
            in_dim = h_dim
            
        self.encoder_body = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        reversed_hidden = hidden_dims[::-1]
        for h_dim in reversed_hidden:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.Dropout(0.2))
            in_dim = h_dim
            
        self.decoder_body = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(hidden_dims[0], input_dim)
        
    def encode(self, x):
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_body(z)
        return torch.sigmoid(self.fc_out(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD, MSE, KLD

def compute_mahalanobis_params(model, train_loader, device='cpu'):
    model.eval()
    mus = []
    with torch.no_grad():
        for (data,) in train_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            mus.append(mu.cpu().numpy())
    mus = np.concatenate(mus, axis=0)
    mean = np.mean(mus, axis=0)
    cov = np.cov(mus, rowvar=False)
    return mean, cov

def mahalanobis_distance(mu, mean, cov_inv):
    diff = mu - mean
    # dist = sqrt(diff * cov_inv * diff.T)
    # Batch computation
    left = np.dot(diff, cov_inv)
    dist = np.sum(left * diff, axis=1)
    return np.sqrt(dist)

def get_hidden_dims(input_dim):
    if input_dim > 100:
        return [256, 128, 64]
    elif input_dim > 50:
        return [128, 64, 32]
    else:
        return [64, 32, 16]

def train_vae(model, train_loader, epochs=50, lr=1e-4, beta=1.0, device='cpu', 
              denoising_noise=0.0, kl_annealing=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        # KL Annealing
        current_beta = beta
        if kl_annealing:
            # Linear annealing from 0 to beta over first 50% of epochs
            anneal_epochs = epochs // 2
            if epoch < anneal_epochs:
                current_beta = beta * (epoch / anneal_epochs)
            else:
                current_beta = beta
        
        total_loss = 0
        mse_loss = 0
        kld_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            
            # Denoising: Add noise to input
            if denoising_noise > 0:
                noisy_data = data + denoising_noise * torch.randn_like(data)
                noisy_data = torch.clamp(noisy_data, 0, 1) # Assuming MinMax scaled [0,1]
            else:
                noisy_data = data
                
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(noisy_data)
            
            # Loss calculated against CLEAN data
            loss, mse, kld = loss_function(recon_batch, data, mu, logvar, beta=current_beta)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            mse_loss += mse.item()
            kld_loss += kld.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            LOGGER.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f} (MSE: {mse_loss/len(train_loader):.6f}, KLD: {kld_loss/len(train_loader):.6f}, Beta: {current_beta:.4f})')
            
    return model

def compute_threshold(model, val_loader, percentile=95, device='cpu', 
                      score_mode='reconstruction', mahalanobis_params=None):
    model.eval()
    scores = []
    
    mean, cov_inv = None, None
    if score_mode == 'mahalanobis' or score_mode == 'combined':
        if mahalanobis_params is None:
            raise ValueError("Mahalanobis params (mean, cov) must be provided for mahalanobis scoring")
        mean, cov = mahalanobis_params
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Add epsilon for stability
            cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
            
    with torch.no_grad():
        for (data,) in val_loader:
            data = data.to(device)
            recon, mu, _ = model(data)
            
            batch_scores = np.zeros(data.shape[0])
            
            # Reconstruction Error
            if score_mode in ['reconstruction', 'combined']:
                loss = nn.functional.mse_loss(recon, data, reduction='none').mean(dim=1)
                batch_scores += loss.cpu().numpy()
                
            # Mahalanobis Distance
            if score_mode in ['mahalanobis', 'combined']:
                mah_dist = mahalanobis_distance(mu.cpu().numpy(), mean, cov_inv)
                if score_mode == 'mahalanobis':
                    batch_scores = mah_dist
                elif score_mode == 'combined':
                    # Very naive combination
                    batch_scores += mah_dist * 0.1 # Weighting guess
            
            scores.extend(batch_scores)
            
    scores = np.array(scores)
    
    # KDE Thresholding
    threshold = np.percentile(scores, percentile)
    LOGGER.info(f"Computed Threshold ({percentile}th percentile, mode={score_mode}): {threshold:.6f}")
    
    return threshold, scores

def plot_kde(errors, threshold, out_path, percentile=95):
    plt.figure(figsize=(10, 6))
    
    # Fixed scale 0-2 as requested
    max_val = 2.0
    
    # Histogram
    plt.hist(errors, bins=100, density=True, alpha=0.6, color='blue', label='Reconstruction Error', range=(0, max_val))
    
    # KDE
    try:
        # Fit KDE on a subset if data is huge for speed
        if len(errors) > 10000:
            kde_errors = np.random.choice(errors, 10000, replace=False)
        else:
            kde_errors = errors
            
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(kde_errors.reshape(-1, 1))
        x_grid = np.linspace(0, max_val, 1000)[:, np.newaxis]
        log_dens = kde.score_samples(x_grid)
        plt.plot(x_grid[:, 0], np.exp(log_dens), color='red', label='KDE')
    except Exception as e:
        LOGGER.warning(f"KDE Plotting failed: {e}")

    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({percentile}th): {threshold:.4f}')
    plt.title(f'Reconstruction Error Distribution')
    plt.xlabel('MSE Loss')
    plt.ylabel('Density')
    plt.xlim(0, max_val)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    plt.close()
    LOGGER.info(f"Saved KDE plot to {out_path}")

def evaluate_vae(model, test_loader, threshold, device='cpu', score_mode='reconstruction', mahalanobis_params=None):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = [] 
    
    mean, cov_inv = None, None
    if score_mode == 'mahalanobis' or score_mode == 'combined':
        if mahalanobis_params is None:
            raise ValueError("Mahalanobis params (mean, cov) must be provided for mahalanobis scoring")
        mean, cov = mahalanobis_params
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            recon, mu, _ = model(data)
            
            batch_scores = np.zeros(data.shape[0])
            
            if score_mode in ['reconstruction', 'combined']:
                loss = nn.functional.mse_loss(recon, data, reduction='none').mean(dim=1)
                batch_scores += loss.cpu().numpy()
                
            if score_mode in ['mahalanobis', 'combined']:
                mah_dist = mahalanobis_distance(mu.cpu().numpy(), mean, cov_inv)
                if score_mode == 'mahalanobis':
                    batch_scores = mah_dist
                elif score_mode == 'combined':
                    batch_scores += mah_dist * 0.1
            
            batch_scores = np.nan_to_num(batch_scores, nan=0.0)
            preds = (batch_scores > threshold).astype(int)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_scores.extend(batch_scores)
            
    return np.array(y_true), np.array(y_pred), np.array(y_scores)

def load_and_process_data(file_paths, label_col='Attack_label', dataset_type='edge-iiot', is_train=False):
    dfs = []
    for path in file_paths:
        LOGGER.info(f"Reading {path}...")
        df = pd.read_csv(path)
        
        # Create Binary Label
        if dataset_type in ['ciciot', 'cic_iomt']:
            if 'label' in df.columns and 'Attack_label' not in df.columns:
                df['Attack_label'] = (df['label'] != 'BenignTraffic').astype(int)
                df.drop(columns=['label'], inplace=True)
        
        # For training, we only want Normal data (Label=0)
        if is_train:
            if label_col in df.columns:
                original_len = len(df)
                df = df[df[label_col] == 0]
                LOGGER.info(f"Filtered Normal data: {len(df)}/{original_len} rows")
        
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Preprocess
    processed_df, pipeline = preprocess_multiclass_data(full_df, label_col=label_col)
    
    return processed_df, pipeline

def main():
    parser = argparse.ArgumentParser(description='Train Phase 1 VAE')
    parser.add_argument('--dataset', type=str, required=True, choices=['edge-iiot', 'ciciot', 'cic_iomt'], help='Dataset name')
    parser.add_argument('--model_path', type=str, help='Path to saved model to skip training')
    parser.add_argument('--threshold_percentile', type=float, default=95.0, help='Percentile for anomaly threshold (default: 95)')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for KL divergence weight (default: 1.0)')
    parser.add_argument('--latent_dim', type=int, default=None, help='Latent dimension size (overrides default)')
    parser.add_argument('--denoising_noise', type=float, default=0.0, help='Noise factor for Denoising VAE (default: 0.0)')
    parser.add_argument('--kl_annealing', action='store_true', help='Enable KL Annealing')
    parser.add_argument('--score_mode', type=str, default='reconstruction', choices=['reconstruction', 'mahalanobis', 'combined'], help='Anomaly scoring mode')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths & Config
    base_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1")
    models_dir = base_dir / "models"
    logs_dir = base_dir / "logs"
    plots_dir = base_dir / "plots"
    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)
    
    if dataset_name == 'edge-iiot':
        data_dir = base_dir / "data/edge_iiot"
        train_files = [data_dir / "normal_training.csv"]
        if not train_files[0].exists():
            LOGGER.warning("normal_training.csv not found, using multiclass_train_80_20.csv (filtered)")
            train_files = [data_dir / "multiclass_train_80_20.csv"]
            
        test_files = [
            data_dir / "multiclass_test_50_50.csv",
            data_dir / "multiclass_test_60_40.csv",
            data_dir / "multiclass_test_70_30.csv",
            data_dir / "multiclass_test_80_20.csv"
        ]
    elif dataset_name == 'ciciot':
        data_dir = base_dir / "data/cic_iot/ciciot_processed_datasets_corrected"
        train_files = [data_dir / "ciciot_training_50_50.csv"]
        test_files = [p for p in data_dir.glob("ciciot_testing_*.csv")]
    elif dataset_name == 'cic_iomt':
        data_dir = base_dir / "data/cici_omt"
        train_files = [data_dir / "cic_iomt_training_50_50.csv"]
        test_files = [p for p in data_dir.glob("cic_iomt_testing_*.csv")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"phase_1_vae_{dataset_name}_{timestamp}"
    LOGGER.info(f"Starting Phase 1 VAE Run: {run_id}")
    
    # 1. Load & Process Training Data (Normal Only)
    LOGGER.info("Loading Training Data (Normal Only)...")
    train_df, pipeline = load_and_process_data(train_files, label_col='Attack_label', dataset_type=dataset_name, is_train=True)
    
    # Drop label for VAE training
    if 'Attack_label' in train_df.columns:
        X_train = train_df.drop(columns=['Attack_label']).values.astype(np.float32)
    else:
        LOGGER.info("Attack_label not found in processed data (likely dropped due to zero variance), using all columns.")
        X_train = train_df.values.astype(np.float32)
        
    X_train = np.nan_to_num(X_train, nan=0.0)
    
    # 2. Build VAE
    input_dim = X_train.shape[1]
    hidden_dims = get_hidden_dims(input_dim)
    if args.latent_dim:
        latent_dim = args.latent_dim
    else:
        latent_dim = min(12, max(3, input_dim // 2))
    
    LOGGER.info(f"Input Dim: {input_dim}, Hidden Dims: {hidden_dims}, Latent Dim: {latent_dim}, Beta: {args.beta}")
    
    model = VAE(input_dim, latent_dim, hidden_dims).to(device)
    
    # 3. Train VAE or Load
    train_dataset = TensorDataset(torch.tensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    if args.model_path:
        LOGGER.info(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        LOGGER.info("Training VAE...")
        model = train_vae(model, train_loader, epochs=50, lr=1e-4, beta=args.beta, device=device,
                          denoising_noise=args.denoising_noise, kl_annealing=args.kl_annealing)
        
        # Save Model
        torch.save(model.state_dict(), models_dir / f"{run_id}_model.pth")
        LOGGER.info("Model Saved.")
    
    # 4. Compute Threshold
    LOGGER.info("Computing Threshold...")
    val_size = int(0.2 * len(X_train))
    val_loader = DataLoader(TensorDataset(torch.tensor(X_train[:val_size])), batch_size=64)
    
    mahalanobis_params = None
    if args.score_mode in ['mahalanobis', 'combined']:
        LOGGER.info("Computing Mahalanobis parameters...")
        mean, cov = compute_mahalanobis_params(model, train_loader, device=device)
        mahalanobis_params = (mean, cov)
        
    threshold, val_errors = compute_threshold(model, val_loader, percentile=args.threshold_percentile, 
                                            device=device, score_mode=args.score_mode, 
                                            mahalanobis_params=mahalanobis_params)
    
    # Plot KDE
    plot_kde(val_errors, threshold, plots_dir / f"{run_id}_kde.png", percentile=args.threshold_percentile)
    
    # 5. Evaluate on Test Sets
    overall_results = {}
    
    for path in test_files:
        name = path.stem
        LOGGER.info(f"Evaluating on {name}...")
        
        test_df = pd.read_csv(path)
        
        # Handle Labels
        if dataset_name in ['ciciot', 'cic_iomt']:
            if 'label' in test_df.columns and 'Attack_label' not in test_df.columns:
                test_df['Attack_label'] = (test_df['label'] != 'BenignTraffic').astype(int)
                test_df.drop(columns=['label'], inplace=True)
        
        if 'Attack_label' not in test_df.columns:
            LOGGER.warning(f"Skipping {name}: Label missing")
            continue
            
        # Preprocess
        df = test_df
        y_test_raw = df['Attack_label'].copy()
        df = df.drop(columns=['Attack_label'])
        
        df = drop_zero_value_features(df)
        df = drop_zero_variance_columns(df)
        df = handle_missing_values(df)
        df = engineer_time_features(df)
        df = apply_onehot_encoding(df)
        df = apply_label_encoding(df)
        df = apply_hash_encoding(df)
        df = convert_object_to_numeric(df)
        
        # Scaling
        if 'minmax_scaler' in pipeline:
            scaler = pipeline['minmax_scaler']
            scaler_features = getattr(scaler, 'feature_names_in_', [])
            for f in scaler_features:
                if f not in df.columns: df[f] = 0
            if len(scaler_features) > 0:
                 df[scaler_features] = scaler.transform(df[scaler_features])
        
        df = apply_log_transformation(df)
        if 'zscore_stats' in pipeline:
            df, _ = apply_zscore_normalization(df, stats=pipeline['zscore_stats'])
        df = create_aggregated_features(df)
        
        # Align columns
        if 'Attack_label' in train_df.columns:
            feature_cols = train_df.drop(columns=['Attack_label']).columns.tolist()
        else:
            feature_cols = train_df.columns.tolist()
        for c in feature_cols:
            if c not in df.columns: df[c] = 0
        df = df[feature_cols]
        
        X_test = df.values.astype(np.float32)
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = y_test_raw.values
        
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        y_true, y_pred, y_scores = evaluate_vae(model, test_loader, threshold, device=device, 
                                              score_mode=args.score_mode, mahalanobis_params=mahalanobis_params)
        
        acc = accuracy_score(y_true, y_pred)
        try:
            auc_score = roc_auc_score(y_true, y_scores)
        except:
            auc_score = 0.0
            
        LOGGER.info(f"{name} Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")
        overall_results[name] = {"accuracy": acc, "auc": auc_score}
        
        # Save Report
        report = classification_report(y_true, y_pred, output_dict=True)
        save_json(report, logs_dir / f"{run_id}_{name}_report.json")
        
        # Plot CM
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, ['Normal', 'Attack'], out_path=plots_dir / f"{run_id}_{name}_cm.png")
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(plots_dir / f"{run_id}_{name}_roc.png")
        plt.close()
        
    LOGGER.info("Overall Results:")
    LOGGER.info(overall_results)
    save_json(overall_results, logs_dir / f"{run_id}_results.json")

if __name__ == "__main__":
    main()
