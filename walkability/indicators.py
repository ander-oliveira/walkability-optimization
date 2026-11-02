from .common import *

def calculate_proximity_weight(time_access: float, max_time: float = 20.0) -> float:
    """
    Calculate proximity weight using a cosine function.
    
    Formula: w(t) = (1 + cos(π×t/20)) / 2
    
    Properties:
    - w(0) = 1 (maximum proximity)
    - w(20) = 0 (accessibility limit)
    - Smooth decay (concave)
    
    Args:
        time_access: Travel time in minutes
        max_time: Maximum acceptable time (default: 20 minutes)
        
    Returns:
        Proximity weight in [0, 1]
    """
    if pd.isna(time_access) or time_access > max_time:
        return 0.0
    return (1 + np.cos(np.pi * time_access / max_time)) / 2

def aggregate_hexagon_indicators(source, t_max: float = 20.0) -> dict:
    """
    Aggregate hexagon indicators into one feature vector.
    """
    _ZERO = {
        'S_saude': 0.0, 'S_educacao': 0.0, 'S_abastecimento': 0.0,
        'S_lazer': 0.0, 'S_servicos': 0.0, 'I_seguranca': 0.0,
        'A_vegetacao': 0.0, 'A_agua': 0.0, 'C_conectividade': 0.0,
        'T_transporte': 0.0, 'U_urbanidade': 0.0
    }

    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str):
        if not os.path.exists(source):
            print(f"⚠ CSV not found: {source}")
            return _ZERO.copy()
        df = pd.read_csv(source)
    else:
        return _ZERO.copy()

    def col(name):
        return df[name] if name in df.columns else pd.Series(0, index=df.index)

    # ── Proximity weights φ(t) — still used for non-POI dimensions ───
    df['w'] = df['min_time_access'].apply(
        lambda t: calculate_proximity_weight(t, max_time=t_max)
    )
    total_weight = df['w'].sum()
    if total_weight == 0:
        return _ZERO.copy()

    # ── Helper: recompute Σα from stored real_times for any t_max ────
    def _alpha_sum_from_times(times_col_name):
        total = 0.0
        for times_str in df[times_col_name].dropna():
            if not times_str or (isinstance(times_str, float) and pd.isna(times_str)):
                continue
            for t_str in str(times_str).split(','):
                t_str = t_str.strip()
                if t_str:
                    try:
                        t = float(t_str)
                        total += calculate_proximity_weight(t, max_time=t_max)
                    except ValueError:
                        continue
        return total

    # ── POI-based dimensions (Solution B: Σα per individual POI) ─────
    _DIM_COLS = {
        'S_saude':         ('alpha_sum_saude',         'times_saude'),
        'S_educacao':      ('alpha_sum_educacao',       'times_educacao'),
        'S_abastecimento': ('alpha_sum_abastecimento',  'times_abastecimento'),
        'S_lazer':         ('alpha_sum_lazer',          'times_lazer'),
        'S_servicos':      ('alpha_sum_servicos',       'times_servicos'),
        'T_transporte':    ('alpha_sum_transporte',     'times_transporte'),
        'U_urbanidade':    ('alpha_sum_urbanidade',     'times_urbanidade'),
    }

    use_precomputed = abs(t_max - 20.0) < 0.01

    poi_results = {}
    for dim_key, (alpha_col, times_col) in _DIM_COLS.items():
        if use_precomputed and alpha_col in df.columns:
            poi_results[dim_key] = col(alpha_col).sum()
        elif times_col in df.columns:
            poi_results[dim_key] = _alpha_sum_from_times(times_col)
        else:
            poi_results[dim_key] = 0.0

    # ── Non-POI dimensions (hexagon-level φ approach) ────────────────
    I_seguranca = (df['w'] * (
        col('count_crosswalks_accessible') +
        col('count_traffic_signals_accessible')
    )).sum()

    A_vegetacao = (df['w'] * col('percent_vegetation')).sum() / total_weight
    A_agua = (df['w'] * col('percent_water')).sum() / total_weight

    C_conectividade = total_weight

    return {
        'S_saude':         round(poi_results['S_saude'], 2),
        'S_educacao':      round(poi_results['S_educacao'], 2),
        'S_abastecimento': round(poi_results['S_abastecimento'], 2),
        'S_lazer':         round(poi_results['S_lazer'], 2),
        'S_servicos':      round(poi_results['S_servicos'], 2),
        'I_seguranca':     round(I_seguranca, 2),
        'A_vegetacao':     round(A_vegetacao, 2),
        'A_agua':          round(A_agua, 2),
        'C_conectividade': round(C_conectividade, 2),
        'T_transporte':    round(poi_results['T_transporte'], 2),
        'U_urbanidade':    round(poi_results['U_urbanidade'], 2),
    }

