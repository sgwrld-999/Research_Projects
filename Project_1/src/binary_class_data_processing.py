from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import zscore
import joblib
from utils import get_logger, ensure_dir

LOGGER = get_logger('binary_data')
LOGGER.output_dir = Path(r"C:\Users\abhay\OneDrive\Desktop\SID\Research_Internship_Under_Dr_Rakesh_Matam\Project_1\logs")


# ============================================================================
# CUSTOM SKLEARN TRANSFORMERS
# ============================================================================

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from DataFrame."""
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.columns].values
        return X[:, self.columns]


class ZeroVarianceDropper(BaseEstimator, TransformerMixin):
    """Drop zero-variance and zero-value columns."""
    def __init__(self):
        self.columns_to_keep_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            # Drop zero-value features
            zero_features = X.columns[(X == 0).all()]
            # Drop zero-variance features
            nunique = X.nunique()
            zero_var = nunique[nunique <= 1].index
            to_drop = set(zero_features) | set(zero_var)
            self.columns_to_keep_ = [col for col in X.columns if col not in to_drop]
            LOGGER.info(f'ZeroVarianceDropper will drop {len(to_drop)} columns')
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.columns_to_keep_ is not None:
            return X[self.columns_to_keep_]
        return X


class DateTimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Convert frame.time to datetime format."""
    def __init__(self, time_column='frame.time'):
        self.time_column = time_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.time_column in X.columns:
            X = X.copy()
            try:
                X[self.time_column] = pd.to_datetime(X[self.time_column], errors='coerce')
                LOGGER.info(f'Converted {self.time_column} to datetime')
            except Exception as e:
                LOGGER.warning(f'Failed to convert {self.time_column}: {e}')
        return X


class BinaryLabelConverter(BaseEstimator, TransformerMixin):
    """Convert multi-class labels to binary (0=Normal, 1=Attack)."""
    def __init__(self, label_col='Attack_label', normal_values=None):
        self.label_col = label_col
        self.normal_values = normal_values or ['Normal', 'normal', 'NORMAL', 'Benign', 'benign', 'BENIGN', 0, '0']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.label_col in X.columns:
            X = X.copy()
            X[f'{self.label_col}_original'] = X[self.label_col].copy()
            X[self.label_col] = X[self.label_col].apply(
                lambda x: 0 if x in self.normal_values else 1
            )
            LOGGER.info(f'Converted to binary labels: {(X[self.label_col] == 0).sum()} Normal, {(X[self.label_col] == 1).sum()} Attack')
        return X


class CorrelationReducer(BaseEstimator, TransformerMixin):
    """Remove highly correlated features."""
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.columns_to_keep_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            numeric_df = X.select_dtypes(include=[np.number])
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr().abs()
                upper_tri = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.threshold)]
                self.columns_to_keep_ = [col for col in X.columns if col not in to_drop]
                LOGGER.info(f'CorrelationReducer will drop {len(to_drop)} highly correlated features')
            else:
                self.columns_to_keep_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.columns_to_keep_ is not None:
            return X[self.columns_to_keep_]
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p transformation to specified features."""
    def __init__(self, features=None):
        self.features = features or ["http.content_length", "mbtcp.len"]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for feature in self.features:
                if feature in X.columns:
                    if not pd.api.types.is_numeric_dtype(X[feature]):
                        X[feature] = pd.to_numeric(X[feature], errors='coerce')
                    X[feature] = np.log1p(X[feature])
            LOGGER.info(f'Applied log transformation to {len([f for f in self.features if f in X.columns])} features')
        return X


class ZScoreNormalizer(BaseEstimator, TransformerMixin):
    """Apply Z-score normalization to numeric columns."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                X[numeric_cols] = X[numeric_cols].apply(zscore)
                LOGGER.info(f'Applied Z-score normalization to {len(numeric_cols)} columns')
        return X


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
# 3. BINARY CLASSIFICATION CONVERSION
# ============================================================================

def convert_to_binary_labels(df: pd.DataFrame, 
                             label_col: str = 'Attack_label',
                             normal_values: List[Union[str, int]] = None) -> pd.DataFrame:
    """
    Convert multi-class attack labels to binary classification (Normal=0, Attack=1).
    
    Args:
        df: Input dataframe
        label_col: Name of the label column
        normal_values: List of values considered as 'Normal'. If None, uses common defaults.
    
    Returns:
        DataFrame with binary labels (0=Normal, 1=Attack)
    """
    if label_col not in df.columns:
        LOGGER.warning(f'Label column "{label_col}" not found in dataframe')
        return df
    
    if normal_values is None:
        # Common representations of normal/benign traffic
        normal_values = ['Normal', 'normal', 'NORMAL', 'Benign', 'benign', 'BENIGN', 0, '0']
    
    # Create binary labels
    original_label_col = f'{label_col}_original'
    df[original_label_col] = df[label_col].copy()
    
    # Convert to binary: 0 for normal, 1 for attack
    df[label_col] = df[label_col].apply(
        lambda x: 0 if x in normal_values else 1
    )
    
    attack_count = (df[label_col] == 1).sum()
    normal_count = (df[label_col] == 0).sum()
    
    LOGGER.info(f'Converted to binary classification: {normal_count} Normal (0), {attack_count} Attack (1)')
    LOGGER.info(f'Original labels preserved in column: {original_label_col}')
    
    return df


def convert_labels_to_binary(labels: np.ndarray, 
                             normal_values: List[Union[str, int]] = None) -> np.ndarray:
    """
    Convert multi-class labels array to binary (0=Normal, 1=Attack).
    
    Args:
        labels: Array of labels
        normal_values: List of values considered as 'Normal'
    
    Returns:
        Binary labels array (0=Normal, 1=Attack)
    """
    if normal_values is None:
        normal_values = ['Normal', 'normal', 'NORMAL', 'Benign', 'benign', 'BENIGN', 0, '0']
    
    binary_labels = np.array([0 if label in normal_values else 1 for label in labels])
    
    attack_count = np.sum(binary_labels == 1)
    normal_count = np.sum(binary_labels == 0)
    
    LOGGER.info(f'Converted labels to binary: {normal_count} Normal (0), {attack_count} Attack (1)')
    
    return binary_labels


# ============================================================================
# 4. FEATURE ENCODING
# ============================================================================

def apply_onehot_encoding(df: pd.DataFrame, 
                         categorical_features: List[str] = None) -> pd.DataFrame:
    """Apply one-hot encoding for low-cardinality categorical features."""
    if categorical_features is None:
        categorical_features = ["tcp.connection.rst", "tcp.connection.syn", 
                               "mqtt.msgtype", "http.response"]
    
    categorical_features_present = [f for f in categorical_features if f in df.columns]
    if categorical_features_present:
        LOGGER.info(f'Applying one-hot encoding to: {categorical_features_present}')
        df = pd.get_dummies(df, columns=categorical_features_present)
    
    return df


def apply_label_encoding(df: pd.DataFrame, 
                        high_cardinality_features: List[str] = None) -> pd.DataFrame:
    """Apply label encoding for high-cardinality features."""
    if high_cardinality_features is None:
        high_cardinality_features = ["dns.qry.type", "mbtcp.trans_id", "mqtt.topic_len"]
    
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
# 5. FEATURE SCALING & NORMALIZATION
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


def apply_zscore_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Z-score normalization to numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Handle NaN values before applying zscore
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df[numeric_cols] = df[numeric_cols].apply(zscore)
        LOGGER.info(f'Applied Z-score normalization to {len(numeric_cols)} numeric columns')
    
    return df


# ============================================================================
# 6. TIME-SERIES FEATURE ENGINEERING
# ============================================================================

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert frame.time column to datetime format."""
    if "frame.time" not in df.columns:
        return df
    
    LOGGER.info('Converting frame.time to datetime format')
    
    try:
        # Convert frame.time to datetime
        df["frame.time"] = pd.to_datetime(df["frame.time"], errors='coerce')
        LOGGER.info('Successfully converted frame.time to datetime')
        
    except Exception as e:
        LOGGER.warning(f'Failed to convert frame.time to datetime: {e}')
    
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
# 7. FEATURE SELECTION & REDUNDANCY REMOVAL
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
# 8. AGGREGATED STATISTICAL FEATURES
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
# 9. OUTLIER DETECTION & REMOVAL
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
# 11. I/O OPERATIONS
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
# 12. COMPLETE PREPROCESSING PIPELINE
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
    
    # Step 1: Convert to binary labels if label column exists
    if label_col in df.columns:
        df = convert_to_binary_labels(df, label_col=label_col)
    
    # Step 2: Handle zero-value and zero-variance features
    df = drop_zero_value_features(df)
    df = drop_zero_variance_columns(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df)
    
    if apply_full_pipeline:
        # Step 4: Time-series feature engineering
        df = engineer_time_features(df)
        
        # Step 5: Feature encoding
        df = apply_onehot_encoding(df)
        df = apply_label_encoding(df)
        df = apply_hash_encoding(df)
        
        # Step 6: Convert remaining objects to numeric
        df = convert_object_to_numeric(df)
        
        # Step 7: Feature scaling
        df, _ = apply_minmax_scaling(df)
        df = apply_log_transformation(df)
        
        # Step 8: Z-score normalization
        df = apply_zscore_normalization(df)
        
        # Step 9: Feature selection
        df = remove_low_variance_features(df, threshold=0.01)
        df = remove_highly_correlated_features(df, correlation_threshold=0.95)
        
        # Step 10: Create aggregated features
        df = create_aggregated_features(df)
        
        # Step 11: Remove outliers
        df = remove_outliers_iqr(df)
    
    LOGGER.info(f'Preprocessing complete. Final shape: {df.shape}')
    
    return df


# ============================================================================
# 13. SKLEARN PIPELINE CREATION
# ============================================================================

def _safe_handle_missing_values(X):
    if isinstance(X, pd.DataFrame):
        return handle_missing_values(X)
    return X

def _safe_apply_encodings(X):
    if isinstance(X, pd.DataFrame):
        return apply_hash_encoding(apply_label_encoding(apply_onehot_encoding(X)))
    return X

def _safe_convert_object_to_numeric(X):
    if isinstance(X, pd.DataFrame):
        return convert_object_to_numeric(X)
    return X

def _safe_apply_minmax_scaling(X):
    if isinstance(X, pd.DataFrame):
        return apply_minmax_scaling(X)[0]
    return X

def _safe_remove_low_variance_features(X, threshold=0.01):
    if isinstance(X, pd.DataFrame):
        return remove_low_variance_features(X, threshold=threshold)
    return X

def _safe_create_aggregated_features(X):
    if isinstance(X, pd.DataFrame):
        return create_aggregated_features(X)
    return X

def _safe_remove_outliers_iqr(X):
    if isinstance(X, pd.DataFrame):
        return remove_outliers_iqr(X)
    return X

def create_preprocessing_pipeline(
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    label_col: str = 'Attack_label'
) -> Pipeline:
    """
    Create a scikit-learn Pipeline for complete data preprocessing.
    
    Args:
        variance_threshold: Threshold for variance-based feature selection
        correlation_threshold: Threshold for correlation-based feature removal
        label_col: Name of the label column
    
    Returns:
        sklearn Pipeline object
    """
    LOGGER.info("Creating sklearn preprocessing pipeline...")
    
    # Define the preprocessing pipeline
    pipeline = Pipeline([
        # Step 1: Convert to binary labels
        ('binary_converter', BinaryLabelConverter(label_col=label_col)),
        
        # Step 2: Drop zero-variance columns
        ('zero_variance_dropper', ZeroVarianceDropper()),
        
        # Step 3: Engineer datetime features
        ('datetime_engineer', DateTimeFeatureEngineer(time_column='frame.time')),
        
        # Step 4: Handle missing values (median for numeric)
        ('imputer', FunctionTransformer(
            _safe_handle_missing_values,
            validate=False
        )),
        
        # Step 5: Feature encoding (one-hot, label, hash)
        ('encoding', FunctionTransformer(
            _safe_apply_encodings,
            validate=False
        )),
        
        # Step 6: Convert objects to numeric
        ('to_numeric', FunctionTransformer(
            _safe_convert_object_to_numeric,
            validate=False
        )),
        
        # Step 7: Log transformation
        ('log_transform', LogTransformer()),
        
        # Step 8: MinMax scaling
        ('minmax_scaling', FunctionTransformer(
            _safe_apply_minmax_scaling,
            validate=False
        )),
        
        # Step 9: Z-score normalization
        ('zscore_normalize', ZScoreNormalizer()),
        
        # Step 10: Remove low variance features
        ('variance_threshold', FunctionTransformer(
            _safe_remove_low_variance_features,
            kw_args={'threshold': variance_threshold},
            validate=False
        )),
        
        # Step 11: Remove highly correlated features
        ('correlation_reducer', CorrelationReducer(threshold=correlation_threshold)),
        
        # Step 12: Create aggregated features
        ('aggregated_features', FunctionTransformer(
            _safe_create_aggregated_features,
            validate=False
        )),
        
        # Step 13: Remove outliers
        ('outlier_removal', FunctionTransformer(
            _safe_remove_outliers_iqr,
            validate=False
        ))
    ])
    
    LOGGER.info("Preprocessing pipeline created successfully")
    return pipeline


def fit_transform_pipeline(
    df: pd.DataFrame,
    pipeline: Pipeline = None,
    save_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Pipeline]:
    """
    Fit and transform data using the preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        pipeline: Pre-created pipeline (if None, creates default)
        save_path: Path to save the fitted pipeline
    
    Returns:
        Tuple of (transformed DataFrame, fitted pipeline)
    """
    LOGGER.info(f"Fitting and transforming pipeline on data with shape: {df.shape}")
    
    if pipeline is None:
        pipeline = create_preprocessing_pipeline()
    
    # Fit and transform
    df_transformed = pipeline.fit_transform(df)
    
    # Save pipeline if path provided
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        joblib.dump(pipeline, save_path)
        LOGGER.info(f"Pipeline saved to: {save_path}")
    
    LOGGER.info(f"Pipeline transformation complete. Output shape: {df_transformed.shape}")
    return df_transformed, pipeline


def load_and_apply_pipeline(
    df: pd.DataFrame,
    pipeline_path: Path
) -> pd.DataFrame:
    """
    Load a saved pipeline and transform new data.
    
    Args:
        df: Input DataFrame
        pipeline_path: Path to the saved pipeline
    
    Returns:
        Transformed DataFrame
    """
    LOGGER.info(f"Loading pipeline from: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    
    LOGGER.info(f"Transforming data with shape: {df.shape}")
    df_transformed = pipeline.transform(df)
    
    LOGGER.info(f"Transformation complete. Output shape: {df_transformed.shape}")
    return df_transformed


def build_phase1_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    val_fraction: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data for Phase-1 VAE training.
    
    Phase-1 VAE is trained on normal traffic for unsupervised anomaly detection.
    This function splits the data into training and validation sets.
    
    Args:
        X: Feature matrix (numpy array) - should contain only normal traffic
        y: Labels (numpy array) - not used, kept for API compatibility
        val_fraction: Fraction of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val)
    """
    from sklearn.model_selection import train_test_split
    
    LOGGER.info(f"Splitting data with shape: {X.shape}")
    
    # Split into train and validation
    X_train, X_val = train_test_split(
        X,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True
    )
    
    LOGGER.info(f"Phase-1 split complete:")
    LOGGER.info(f"  Training set: {X_train.shape}")
    LOGGER.info(f"  Validation set: {X_val.shape}")
    
    return X_train, X_val



