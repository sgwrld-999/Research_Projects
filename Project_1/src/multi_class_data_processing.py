from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import zscore
from utils import get_logger, ensure_dir

LOGGER = get_logger('binary_data')
LOGGER.output_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\logs")


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_csv(path: str, low_memory: bool = False) -> pd.DataFrame:
    """Load CSV file with logging."""
    LOGGER.info(f'Loading CSV: {path}')
    df = pd.read_csv(path, low_memory=low_memory)
    LOGGER.info(f'Loaded shape: {df.shape}')
    return df


def load_normal_train(base_path) -> pd.DataFrame:
    """Load normal training data. Accepts str or Path."""
    path = Path(base_path) / 'normal_training.csv'
    return load_csv(str(path))


def load_normal_test(base_path) -> pd.DataFrame:
    """Load normal testing data. Accepts str or Path."""
    path = Path(base_path) / 'normal_testing.csv'
    return load_csv(str(path))


# ============================================================================
# 2. MISSING VALUES & ZERO VARIANCE
# ============================================================================

def drop_zero_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop features with only zero values."""
    zero_variance_features = df.columns[(df == 0).all()]
    if len(zero_variance_features) > 0:
        LOGGER.info(f'Dropping zero-value features: {list(zero_variance_features)}')
        df = df.drop(columns=zero_variance_features)
    else:
        LOGGER.info('No zero-value features to drop')
    return df


def drop_zero_variance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with zero variance (single unique value)."""
    nunique = df.nunique()
    to_drop = nunique[nunique <= 1].index.tolist()
    if to_drop:
        LOGGER.info(f'Dropping zero variance columns: {to_drop}')
        df = df.drop(columns=to_drop)
    else:
        LOGGER.info('No zero variance columns to drop')
    return df


def separate_numeric_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Separate numeric and categorical column names."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    LOGGER.info(f'Numeric columns: {len(numeric_cols)}, Categorical columns: {len(categorical_cols)}')
    return numeric_cols, categorical_cols


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply median imputation for numeric and mode for categorical columns."""
    numeric_cols, categorical_cols = separate_numeric_categorical(df)
    
    # Median imputation for numeric
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        LOGGER.info(f'Applied median imputation to {len(numeric_cols)} numeric columns')
    
    # Mode imputation for categorical
    if categorical_cols:
        for col in categorical_cols:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
        LOGGER.info(f'Applied mode imputation to {len(categorical_cols)} categorical columns')
    
    return df


# ============================================================================
# 3. FEATURE ENCODING
# ============================================================================

def apply_onehot_encoding(df: pd.DataFrame, 
                         categorical_features: List[str] = None) -> pd.DataFrame:
    """Apply one-hot encoding for low-cardinality categorical features."""
    if categorical_features is None:
        categorical_features = ["tcp.connection.rst", "tcp.connection.syn", 
                               "mqtt.msgtype", "http.response", "http.request.method"]
    
    categorical_features_present = [f for f in categorical_features if f in df.columns]
    if categorical_features_present:
        LOGGER.info(f'Applying one-hot encoding to: {categorical_features_present}')
        df = pd.get_dummies(df, columns=categorical_features_present)
    
    return df


def apply_label_encoding(df: pd.DataFrame, 
                        high_cardinality_features: List[str] = None) -> pd.DataFrame:
    """Apply label encoding for high-cardinality features."""
    if high_cardinality_features is None:
        high_cardinality_features = ["dns.qry.type", "mbtcp.trans_id", "mqtt.topic_len", "http.referer", "http.user_agent", "http.agent"]
    
    for feature in high_cardinality_features:
        if feature in df.columns:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature].astype(str))
            LOGGER.info(f'Applied label encoding to: {feature}')
    
    return df


def apply_hash_encoding(df: pd.DataFrame, 
                       hash_features: List[str] = None, 
                       hash_bins: int = 10000) -> pd.DataFrame:
    """Apply hash encoding for IP addresses and domain names."""
    if hash_features is None:
        hash_features = ["ip.src_host", "ip.dst_host", "dns.qry.name"]
    
    for feature in hash_features:
        if feature in df.columns:
            df[feature] = df[feature].apply(lambda x: hash(str(x)) % hash_bins)
            LOGGER.info(f'Applied hash encoding to: {feature}')
    
    return df


# ============================================================================
# 4. FEATURE SCALING & NORMALIZATION
# ============================================================================

def apply_minmax_scaling(df: pd.DataFrame, 
                        min_max_features: List[str] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Apply Min-Max scaling to sequential numeric features."""
    if min_max_features is None:
        min_max_features = ["tcp.seq", "udp.time_delta", "icmp.transmit_timestamp"]
    
    min_max_features_present = [f for f in min_max_features if f in df.columns]
    scaler = None
    
    if min_max_features_present:
        scaler = MinMaxScaler()
        df[min_max_features_present] = scaler.fit_transform(df[min_max_features_present])
        LOGGER.info(f'Applied MinMax scaling to: {min_max_features_present}')
    
    return df, scaler


def apply_log_transformation(df: pd.DataFrame, 
                             log_transform_features: List[str] = None) -> pd.DataFrame:
    """Apply log transformation for skewed features."""
    if log_transform_features is None:
        log_transform_features = ["http.content_length", "mbtcp.len"]
    
    for feature in log_transform_features:
        if feature in df.columns:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
            
            df[feature] = np.log1p(df[feature])  # log(1 + x) to avoid log(0)
            LOGGER.info(f'Applied log transformation to: {feature}')
    
    return df


def apply_zscore_normalization(df: pd.DataFrame, stats: Dict[str, Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Apply Z-score normalization to numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    current_stats = {}
    
    if len(numeric_cols) > 0:
        # Handle NaN values before applying zscore
        # If stats provided, we assume NaNs handled or we use provided mean?
        # For consistency, we fillna with mean.
        
        if stats is None:
            # Fit and transform
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std(ddof=0) # Use population std or sample? zscore uses population by default? No, ddof=0 usually.
                # scipy.stats.zscore uses ddof=0 by default.
                
                if std == 0:
                    std = 1.0
                
                df[col] = df[col].fillna(mean)
                df[col] = (df[col] - mean) / std
                current_stats[col] = {'mean': mean, 'std': std}
            LOGGER.info(f'Applied Z-score normalization to {len(numeric_cols)} numeric columns')
        else:
            # Transform using provided stats
            count = 0
            for col in numeric_cols:
                if col in stats:
                    mean = stats[col]['mean']
                    std = stats[col]['std']
                    df[col] = df[col].fillna(mean)
                    df[col] = (df[col] - mean) / std
                    count += 1
            current_stats = stats
            LOGGER.info(f'Applied Z-score normalization using provided stats to {count} columns')
    
    return df, current_stats


# ============================================================================
# 5. TIME-SERIES FEATURE ENGINEERING
# ============================================================================

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from frame.time column."""
    if "frame.time" not in df.columns:
        return df
    
    LOGGER.info('Engineering time-based features from frame.time')
    
    # Ensure frame.time is string type
    df["frame.time"] = df["frame.time"].astype(str)
    
    # Extract year
    df["frame.time_year"] = df["frame.time"].str.extract(r'(\d{4})').astype(float)
    
    # Extract hour
    df["frame.time_hour"] = df["frame.time"].str.extract(r'(\d{2}):\d{2}:\d{2}').astype(float)
    
    # Extract minute
    df["frame.time_minute"] = df["frame.time"].str.extract(r'\d{2}:(\d{2}):\d{2}').astype(float)
    
    # Extract second
    df["frame.time_second"] = df["frame.time"].str.extract(r'\d{2}:\d{2}:(\d{2})').astype(float)
    
    # Drop the original frame.time column
    df.drop(columns=["frame.time"], inplace=True)
    
    LOGGER.info('Extracted time features: year, hour, minute, second')
    
    return df


def convert_object_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert remaining object columns to numeric where possible."""
    object_cols = df.select_dtypes(include=['object']).columns
    
    if len(object_cols) > 0:
        for col in object_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        LOGGER.info(f'Converted {len(object_cols)} object columns to numeric')
    
    return df


# ============================================================================
# 6. FEATURE SELECTION & REDUNDANCY REMOVAL
# ============================================================================

def remove_low_variance_features(df: pd.DataFrame, 
                                threshold: float = 0.01) -> pd.DataFrame:
    """Remove low-variance features using VarianceThreshold."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        LOGGER.warning('No numeric columns for variance threshold')
        return df
    
    # Handle NaN values
    numeric_df = numeric_df.fillna(0)
    
    selector = VarianceThreshold(threshold=threshold)
    numeric_df_transformed = pd.DataFrame(
        selector.fit_transform(numeric_df), 
        columns=numeric_df.columns[selector.get_support()]
    )
    
    removed_count = len(numeric_df.columns) - len(numeric_df_transformed.columns)
    LOGGER.info(f'Removed {removed_count} low-variance features (threshold={threshold})')
    
    # Reconstruct DataFrame
    categorical_df = df.select_dtypes(exclude=[np.number])
    df = pd.concat([numeric_df_transformed, categorical_df], axis=1)
    
    return df


def remove_highly_correlated_features(df: pd.DataFrame, 
                                      correlation_threshold: float = 0.95) -> pd.DataFrame:
    """Remove highly correlated features."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        LOGGER.warning('Not enough numeric columns for correlation analysis')
        return df
    
    correlation_matrix = numeric_df.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    high_correlation_features = [
        column for column in upper_tri.columns 
        if any(upper_tri[column] > correlation_threshold)
    ]
    
    if high_correlation_features:
        df = df.drop(columns=high_correlation_features)
        LOGGER.info(f'Removed {len(high_correlation_features)} highly correlated features (threshold={correlation_threshold})')
    
    return df


# ============================================================================
# 7. AGGREGATED STATISTICAL FEATURES
# ============================================================================

def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated statistical features grouped by IP."""
    if "ip.src_host" not in df.columns:
        return df
    
    LOGGER.info('Creating aggregated statistical features')
    
    if "tcp.connection.rst" in df.columns:
        df["tcp.rst_count"] = df.groupby("ip.src_host")["tcp.connection.rst"].transform("sum")
        LOGGER.info('Created tcp.rst_count feature')
    
    if "mqtt.topic_len" in df.columns and pd.api.types.is_numeric_dtype(df["mqtt.topic_len"]):
        df["mqtt.topic_unique"] = df.groupby("ip.src_host")["mqtt.topic_len"].transform("nunique")
        LOGGER.info('Created mqtt.topic_unique feature')
    
    return df


# ============================================================================
# 8. OUTLIER DETECTION & REMOVAL
# ============================================================================

def remove_outliers_iqr(df: pd.DataFrame, 
                       outlier_features: List[str] = None,
                       iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    if outlier_features is None:
        outlier_features = ["http.content_length", "tcp.seq"]
    
    outlier_features_present = [f for f in outlier_features if f in df.columns]
    initial_shape = df.shape[0]
    
    for feature in outlier_features_present:
        if pd.api.types.is_numeric_dtype(df[feature]):
            Q1, Q3 = np.percentile(df[feature], [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    
    removed_count = initial_shape - df.shape[0]
    if removed_count > 0:
        LOGGER.info(f'Removed {removed_count} outlier rows using IQR method')
    
    return df


# ============================================================================
# 9. PHASE-1 SPECIFIC FUNCTIONS
# ============================================================================

def build_phase1_split(X: np.ndarray, 
                      y: Optional[np.ndarray] = None, 
                      val_fraction: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Split data for Phase-1 (benign samples only for VAE training)."""
    if y is None:
        LOGGER.warning('No labels provided to build_phase1_split; returning full X as train set')
        return X, np.empty((0, X.shape[1]))
    
    # Normalize label checking
    mask_benign = (y == 0) | (y == 'Normal') | (y == 'normal')
    X_ben = X[mask_benign]
    LOGGER.info(f'Phase-1: benign samples count {X_ben.shape[0]}')
    
    n_val = int(len(X_ben) * val_fraction)
    if n_val == 0:
        return X_ben, np.empty((0, X.shape[1]))
    
    X_train = X_ben[:-n_val]
    X_val = X_ben[-n_val:]
    LOGGER.info(f'Phase-1 split train {X_train.shape}, val {X_val.shape}')
    
    return X_train, X_val


# ============================================================================
# 10. I/O OPERATIONS
# ============================================================================

def save_npz(path, **arrays):
    """Save numpy arrays to compressed npz file. Accepts str or Path."""
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    LOGGER.info(f'Saved npz to {path}')


def load_npz(path):
    """Load numpy arrays from npz file. Accepts str or Path."""
    path = Path(path)
    LOGGER.info(f'Loading npz {path}')
    return np.load(path)


# ============================================================================
# 11. COMPLETE PREPROCESSING PIPELINE
# ============================================================================

def preprocess_binary_data(df: pd.DataFrame, 
                          label_col: str = 'Attack_label',
                          apply_full_pipeline: bool = True) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for binary classification data.
    
    Args:
        df: Input dataframe
        label_col: Name of label column
        apply_full_pipeline: If True, apply all preprocessing steps
    
    Returns:
        Preprocessed dataframe
    """
    LOGGER.info(f'Starting preprocessing pipeline. Initial shape: {df.shape}')
    
    # Step 1: Handle zero-value and zero-variance features
    df = drop_zero_value_features(df)
    df = drop_zero_variance_columns(df)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    if apply_full_pipeline:
        # Step 3: Time-series feature engineering
        df = engineer_time_features(df)
        
        # Step 4: Feature encoding
        df = apply_onehot_encoding(df)
        df = apply_label_encoding(df)
        df = apply_hash_encoding(df)
        
        # Step 5: Convert remaining objects to numeric
        df = convert_object_to_numeric(df)
        
        # Step 6: Feature scaling
        df, _ = apply_minmax_scaling(df)
        df = apply_log_transformation(df)
        
        # Step 7: Z-score normalization
        df, _ = apply_zscore_normalization(df)
        
        # Step 8: Feature selection
        df = remove_low_variance_features(df, threshold=0.01)
        df = remove_highly_correlated_features(df, correlation_threshold=0.95)
        
        # Step 9: Create aggregated features
        df = create_aggregated_features(df)
        
        # Step 10: Remove outliers
        df = remove_outliers_iqr(df)
    
    LOGGER.info(f'Preprocessing complete. Final shape: {df.shape}')
    
    return df




def preprocess_multiclass_data(df: pd.DataFrame, 
                               label_col: str = 'Attack_type',
                               apply_full_pipeline: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline for multi-class classification data.
    
    Args:
        df: Input dataframe
        label_col: Name of label column (default: 'Attack_type')
        apply_full_pipeline: If True, apply all preprocessing steps
    
    Returns:
        Tuple of (Preprocessed dataframe, Pipeline dictionary containing fitted objects)
    """
    LOGGER.info(f'Starting multi-class preprocessing pipeline. Initial shape: {df.shape}')
    pipeline = {}
    
    # Step 0: Encode Labels first to ensure we don't lose them
    le = LabelEncoder()
    if label_col in df.columns:
        # Filter out 'Attack_label' if present as it's for binary, UNLESS it is the target
        if 'Attack_label' in df.columns and label_col != 'Attack_label':
            df = df.drop(columns=['Attack_label'])
            
        df[label_col] = le.fit_transform(df[label_col].astype(str))
        pipeline['le'] = le
        LOGGER.info(f'Encoded labels for {label_col}. Classes: {le.classes_}')
    else:
        LOGGER.warning(f'Label column {label_col} not found!')
    
    # Step 1: Handle zero-value and zero-variance features
    df = drop_zero_value_features(df)
    df = drop_zero_variance_columns(df)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    if apply_full_pipeline:
        # Step 3: Time-series feature engineering
        df = engineer_time_features(df)
        
        # Step 4: Feature encoding
        # Exclude label_col from encoding
        df = apply_onehot_encoding(df)
        df = apply_label_encoding(df)
        df = apply_hash_encoding(df)
        
        # Step 5: Convert remaining objects to numeric
        df = convert_object_to_numeric(df)
        
        # Step 6: Feature scaling
        df, minmax_scaler = apply_minmax_scaling(df)
        pipeline['minmax_scaler'] = minmax_scaler
        
        df = apply_log_transformation(df)
        
        # Step 7: Z-score normalization
        # Exclude label column from normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col in numeric_cols:
            numeric_cols.remove(label_col)
            
        if len(numeric_cols) > 0:
            # We need to pass only numeric cols to apply_zscore_normalization or it handles it?
            # It handles it but modifies in place.
            # We want to normalize ONLY feature columns, not label.
            # But apply_zscore_normalization selects all numeric.
            # So we should temporarily drop label or pass subset?
            # Or modify apply_zscore_normalization to accept columns?
            # It selects all numeric.
            # So we can drop label, normalize, add label back.
            
            if label_col in df.columns:
                y = df[label_col].copy()
                df_features = df.drop(columns=[label_col])
                
                df_features, zscore_stats = apply_zscore_normalization(df_features)
                pipeline['zscore_stats'] = zscore_stats
                
                df = pd.concat([df_features, y], axis=1)
                LOGGER.info(f'Applied Z-score normalization (excluding label)')
            else:
                df, zscore_stats = apply_zscore_normalization(df)
                pipeline['zscore_stats'] = zscore_stats
                LOGGER.info(f'Applied Z-score normalization (label not found)')
        
        # Step 8: Feature selection
        # Note: We skip aggressive feature selection for now to ensure consistency across datasets
        # or we should save the selected columns. For this pipeline, we'll keep it simple.
        
        # Step 9: Create aggregated features
        df = create_aggregated_features(df)
        
        # Step 10: Remove outliers (Optional - can be risky for minority classes)
        # df = remove_outliers_iqr(df) 
    
    LOGGER.info(f'Multi-class preprocessing complete. Final shape: {df.shape}')
    
    return df, pipeline
