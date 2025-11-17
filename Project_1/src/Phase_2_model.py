from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils import get_logger, ensure_dir

LOGGER = get_logger('phase2_model')

# Set device for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER.info(f'Using device: {device}')

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_shape, num_classes, lstm_units=32, dropout_rate=0.3):
        super(CNN_BiLSTM, self).__init__()
        # input_shape: (seq_len, num_features)
        # Conv1d expects (batch, channels, seq_len)
        
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # LSTM
        # Input size is out_channels of previous layer = 128
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_units, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(lstm_units * 2, 64) # Bidirectional * hidden_size
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc_out = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm(x)
        
        # Global Average Pooling on LSTM output
        # x shape: (batch, seq_len, hidden*2)
        x = x.permute(0, 2, 1) # (batch, hidden*2, seq_len)
        x = self.global_avg_pool(x).squeeze(-1) # (batch, hidden*2)
        
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        x = self.fc_out(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    n_samples, n_feats = X.shape
    if n_samples < seq_len:
        raise ValueError('Not enough samples to build even a single sequence')
    
    # Simple sliding window
    # X_seq: (n_samples - seq_len + 1, seq_len, n_feats)
    # y_seq: (n_samples - seq_len + 1,)
    
    # Optimized way using stride_tricks or just loop if memory allows
    # For clarity and safety with labels, loop is fine or list comp
    
    X_seq = []
    y_seq = []
    
    # We take the label of the last element in the sequence
    for i in range(n_samples - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    LOGGER.info(f'Built sequences shape: {X_seq.shape} from X shape {X.shape} with seq_len={seq_len}')
    return X_seq, y_seq

from data_augmentation import augment_minority_classes

def train_phase2(X_train, y_train, seq_len, num_classes, run_dir, epochs=30, batch_size=256, class_weights=None, lstm_units=32, dropout_rate=0.3, learning_rate=1e-3, augment_classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")
    
    # Create sequences
    X_seq, y_seq = build_sequences(X_train, y_train, seq_len)
    
    # Split validation
    X_t, X_v, y_t, y_v = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
    
    # Apply Augmentation
    if augment_classes:
        LOGGER.info(f"Applying augmentation to classes: {augment_classes}")
        X_t, y_t = augment_minority_classes(X_t, y_t, target_classes=augment_classes, multiplier=1)
    
    # Convert to tensors
    X_t = torch.tensor(X_t, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_t, dtype=torch.long).to(device)
    X_v = torch.tensor(X_v, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_v, dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(X_t, y_t)
    val_dataset = TensorDataset(X_v, y_v)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = CNN_BiLSTM(input_shape=(seq_len, X_train.shape[1]), num_classes=num_classes, lstm_units=lstm_units, dropout_rate=dropout_rate).to(device)
    
    # Loss with class weights
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        LOGGER.info(f"Using Class Weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
        epoch_loss = train_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        LOGGER.info(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {val_epoch_loss:.4f} - val_acc: {val_epoch_acc:.4f}")
        
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
            ensure_dir(run_dir)
            torch.save(model.state_dict(), run_dir / 'best_model.pth')
            LOGGER.info(f"Saved best model to {run_dir / 'best_model.pth'}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                LOGGER.info("Early stopping triggered")
                break
                
    # Load best model
    model.load_state_dict(torch.load(run_dir / 'best_model.pth'))
    return model, history

def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, seq_len: int = 10, batch_size: int = 256):
    """Evaluate model on test data."""
    model.eval()
    
    # Build sequences
    # Note: build_sequences expects y to be present.
    X_seq, y_seq = build_sequences(X, y, seq_len)
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq, dtype=torch.long).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)
