from .common import *


HEX_TIME_MATRIX_COLUMNS = [
    'source_h3_id',
    'target_h3_id',
    'source_node_id',
    'target_node_id',
    'time_min',
    'alpha_20',
    'profile_key',
]


def calculate_time_decay_weight(time_min: float, max_time: float = 20.0) -> float:
    """Cosine-decay weight for walking time."""
    if pd.isna(time_min):
        return 0.0

    t = float(time_min)
    if t < 0 or t > max_time:
        return 0.0

    return float((1 + np.cos(np.pi * t / max_time)) / 2)


def build_hex_time_matrix(graph,
                          df_walkability_base: pd.DataFrame,
                          profile: dict,
                          profile_key: str = None,
                          max_time: float = 20.0) -> pd.DataFrame:
    """
    Build a source-target hexagon time/impact matrix using graph edge time (Tobler-aware).

    Output keeps only pairs with total travel time <= max_time.
    """
    if graph is None:
        raise ValueError("graph is required to build hex time matrix.")

    if df_walkability_base is None or df_walkability_base.empty:
        return pd.DataFrame(columns=HEX_TIME_MATRIX_COLUMNS)

    required_cols = ['h3_id', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df_walkability_base.columns]
    if missing_cols:
        raise ValueError(f"df_walkability_base is missing required columns: {missing_cols}")

    df_hex = (
        df_walkability_base[required_cols]
        .copy()
        .dropna(subset=required_cols)
        .drop_duplicates(subset=['h3_id'], keep='first')
    )
    if df_hex.empty:
        return pd.DataFrame(columns=HEX_TIME_MATRIX_COLUMNS)

    df_hex['h3_id'] = df_hex['h3_id'].astype(str)

    if 'crs' not in graph.graph:
        raise ValueError("graph has no CRS metadata (graph.graph['crs']).")

    gdf_hex = gpd.GeoDataFrame(
        df_hex.copy(),
        geometry=gpd.points_from_xy(df_hex['longitude'], df_hex['latitude']),
        crs='EPSG:4326',
    ).to_crs(graph.graph['crs'])

    nearest_nodes = ox.distance.nearest_nodes(
        graph,
        gdf_hex.geometry.x.to_numpy(),
        gdf_hex.geometry.y.to_numpy(),
    )
    nearest_nodes = list(nearest_nodes) if hasattr(nearest_nodes, '__iter__') else [nearest_nodes]

    speed_walk_kmh = float(profile.get('speed_walk', 5.0))
    speed_walk_m_per_min = max(speed_walk_kmh * 1000.0 / 60.0, 0.001)

    node_x = nx.get_node_attributes(graph, 'x')
    node_y = nx.get_node_attributes(graph, 'y')

    mapping_rows = []
    for idx, row in enumerate(gdf_hex.itertuples(index=False)):
        node_id = nearest_nodes[idx]
        nx_x = node_x.get(node_id)
        nx_y = node_y.get(node_id)

        if nx_x is None or nx_y is None:
            access_time = 0.0
        else:
            access_distance_m = row.geometry.distance(Point(nx_x, nx_y))
            access_time = float(access_distance_m / speed_walk_m_per_min)

        mapping_rows.append({
            'h3_id': row.h3_id,
            'node_id': node_id,
            'access_time_min': access_time,
        })

    df_mapping = pd.DataFrame(mapping_rows)
    if df_mapping.empty:
        return pd.DataFrame(columns=HEX_TIME_MATRIX_COLUMNS)

    unique_source_nodes = df_mapping['node_id'].dropna().unique().tolist()
    if not unique_source_nodes:
        return pd.DataFrame(columns=HEX_TIME_MATRIX_COLUMNS)

    node_paths = {}
    for source_node in unique_source_nodes:
        node_paths[source_node] = nx.single_source_dijkstra_path_length(
            graph,
            source_node,
            weight='time',
            cutoff=max_time,
        )

    profile_tag = profile_key or graph.graph.get('_time_profile_key') or profile.get('name', 'unknown_profile')

    targets = list(df_mapping.itertuples(index=False))
    output_rows = []
    for source in df_mapping.itertuples(index=False):
        source_paths = node_paths.get(source.node_id, {})
        for target in targets:
            path_time = source_paths.get(target.node_id)
            if path_time is None:
                continue

            total_time = float(source.access_time_min + path_time + target.access_time_min)
            if total_time > max_time:
                continue

            alpha = calculate_time_decay_weight(total_time, max_time=max_time)
            if alpha <= 0:
                continue

            output_rows.append({
                'source_h3_id': source.h3_id,
                'target_h3_id': target.h3_id,
                'source_node_id': source.node_id,
                'target_node_id': target.node_id,
                'time_min': round(total_time, 4),
                'alpha_20': round(alpha, 6),
                'profile_key': profile_tag,
            })

    if not output_rows:
        return pd.DataFrame(columns=HEX_TIME_MATRIX_COLUMNS)

    df_matrix = pd.DataFrame(output_rows)
    df_matrix = (
        df_matrix.sort_values(['source_h3_id', 'target_h3_id', 'time_min'])
        .drop_duplicates(subset=['source_h3_id', 'target_h3_id'], keep='first')
        .reset_index(drop=True)
    )
    return df_matrix
