from .common import *

def apply_critic_method(df: pd.DataFrame) -> pd.Series:
    """
    Applies CRITIC method to calculate objective weights.
    
    CRITIC (CRiteria Importance Through Intercriteria Correlation) determines
    weights based on:
    1. Contrast: Standard deviation (higher variability = more informative)
    2. Conflict: Low correlation with other indicators (less redundancy)
    
    Mathematical formulation:
    - C_j = σ_j × Σ(1 - r_jk) where σ_j is std dev and r_jk is correlation
    - w_j = C_j / Σ(C_j)
    
    Args:
        df: DataFrame with RAW (non-normalized) indicators
        
    Returns:
        Series with weights for each indicator (sum = 1)
    """
    # Min-Max Normalization for standard deviation calculation
    df_range = df.max() - df.min()
    
    # Check for zero variance indicators
    zero_var_cols = df_range[df_range == 0].index.tolist()
    if zero_var_cols:
        print(f"⚠ Warning: Zero variance indicators will receive zero weight: {zero_var_cols}")
    
    df_norm = df.copy()
    for col in df.columns:
        if df_range[col] > 0:
            df_norm[col] = (df[col] - df[col].min()) / df_range[col]
        else:
            df_norm[col] = 0.0  # Zero variance = no information
    
    # Standard deviation (contrast) - on normalized data
    std_dev = df_norm.std()
    
    # Correlation matrix - on RAW data (correlation is scale-invariant)
    # Only compute for non-zero variance indicators
    valid_cols = [col for col in df.columns if df_range[col] > 0]
    
    if len(valid_cols) < 2:
        # Not enough indicators to compute correlation, use uniform weights
        W = pd.Series(1.0 / len(df.columns), index=df.columns)
        return W
    
    corr_matrix = df[valid_cols].corr().abs()
    
    # Information quantity (conflict)
    conflict = pd.Series(0.0, index=df.columns)
    conflict[valid_cols] = (1 - corr_matrix).sum(axis=1)
    
    # CRITIC measure
    C = std_dev * conflict
    
    # Normalized weights
    C_sum = C.sum()
    if C_sum > 0:
        W = C / C_sum
    else:
        # Fallback: uniform weights
        W = pd.Series(1.0 / len(df.columns), index=df.columns)
    
    return W

def calculate_walkability_index(df_indicators: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute IQC with CRITIC weights and additive aggregation."""
    id_cols = ['h3_id', 'latitude', 'longitude']
    indicator_cols = [col for col in df_indicators.columns if col not in id_cols]

    weights = apply_critic_method(df_indicators[indicator_cols])
    df_norm = pd.DataFrame(index=df_indicators.index, columns=indicator_cols, dtype=float)

    for col in indicator_cols:
        col_min = df_indicators[col].min()
        col_max = df_indicators[col].max()
        col_range = col_max - col_min
        if col_range > 0:
            df_norm[col] = (df_indicators[col] - col_min) / col_range
        else:
            df_norm[col] = 0.5

    df_indicators['IQC'] = (df_norm * weights).sum(axis=1)
    df_indicators['IQC'] = df_indicators['IQC'].round(4)
    return df_indicators, weights

def compute_walkability_for_all_hexagons(location: str,
                                        profile_key: str,
                                        h3_resolution: int,
                                        base_dir: str = 'data',
                                        df_indicators: pd.DataFrame = None,
                                        distance: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute IQC from an in-memory indicators dataframe."""
    if df_indicators is None or df_indicators.empty:
        print('No indicators dataframe available for IQC computation.')
        return pd.DataFrame(), pd.Series(dtype=float)

    df_final, critic_weights = calculate_walkability_index(df_indicators)
    print(f'IQC computed for {len(df_final)} hexagons.')
    return df_final, critic_weights

