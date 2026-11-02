from .common import *

def save_core_dataframes(df_walkability: pd.DataFrame,
                         critic_weights: pd.Series,
                         df_indicators_base: pd.DataFrame,
                         df_hex_time_matrix: pd.DataFrame,
                         location: str,
                         profile_key: str,
                         h3_resolution: int,
                         base_dir: str = 'data',
                         distance: int = None,
                         save_walkability: bool = True,
                         save_critic_weights: bool = True,
                         save_indicators_base: bool = True,
                         save_hex_time_matrix: bool = True) -> dict:
    """Persist core dataframes used by the metaheuristic entry point."""
    profile_suffix = f"_{profile_key}" if profile_key else ""
    dist_suffix = f"_dist{distance}" if distance is not None else ""
    out_dir = os.path.join(base_dir, 'csv', 'walkability_index')
    os.makedirs(out_dir, exist_ok=True)

    paths = {}

    if save_walkability and df_walkability is not None and not df_walkability.empty:
        walkability_file = os.path.join(
            out_dir,
            f"{location}_walkability_index_{profile_key}_res_{h3_resolution}{dist_suffix}.csv"
        )
        df_walkability.to_csv(walkability_file, index=False, encoding='utf-8')
        paths['df_walkability'] = walkability_file

    if save_critic_weights and critic_weights is not None and len(critic_weights) > 0:
        weights_file = os.path.join(
            out_dir,
            f"{location}_critic_weights{profile_suffix}_res_{h3_resolution}{dist_suffix}.csv"
        )
        df_weights = critic_weights.rename('weight').reset_index().rename(columns={'index': 'indicator'})
        df_weights.to_csv(weights_file, index=False, encoding='utf-8')
        paths['critic_weights'] = weights_file

    if save_indicators_base and df_indicators_base is not None and not df_indicators_base.empty:
        indicators_file = os.path.join(
            out_dir,
            f"{location}_df_indicators_base{profile_suffix}_res_{h3_resolution}{dist_suffix}.csv"
        )
        df_indicators_base.to_csv(indicators_file, index=False, encoding='utf-8')
        paths['df_indicators_base'] = indicators_file

    if save_hex_time_matrix and df_hex_time_matrix is not None and not df_hex_time_matrix.empty:
        hex_time_matrix_file = os.path.join(
            out_dir,
            f"{location}_hex_time_matrix_{profile_key}_res_{h3_resolution}{dist_suffix}.csv"
        )
        df_hex_time_matrix.to_csv(hex_time_matrix_file, index=False, encoding='utf-8')
        paths['hex_time_matrix'] = hex_time_matrix_file

    print(f'Saved {len(paths)} core dataframe file(s).')
    return paths

