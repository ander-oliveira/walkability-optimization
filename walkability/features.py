from .common import *

def map_poi_colors_and_types(gdf_pois: gpd.GeoDataFrame, 
                            color_map: dict) -> gpd.GeoDataFrame:

    if gdf_pois.empty: 
        return gdf_pois
    
    # OSM tags like building=yes / building=no are generic and carry no useful type info.
    # Replace them with NaN so the fallback chain picks the more specific tag instead.
    _generic = ['yes', 'no']
    amenity_col  = gdf_pois['amenity'].replace(_generic, pd.NA)  if 'amenity'  in gdf_pois.columns else pd.Series(pd.NA, index=gdf_pois.index)
    building_col = gdf_pois['building'].replace(_generic, pd.NA) if 'building' in gdf_pois.columns else pd.Series(pd.NA, index=gdf_pois.index)
    shop_col     = gdf_pois['shop'].replace(_generic, pd.NA)     if 'shop'     in gdf_pois.columns else pd.Series(pd.NA, index=gdf_pois.index)

    gdf_pois['poi_type'] = amenity_col.fillna(building_col.fillna(shop_col))
    gdf_pois['color'] = gdf_pois['poi_type'].map(color_map).fillna('grey')
    
    return gdf_pois

def calculate_feature_travel_time(gdf_features: gpd.GeoDataFrame, graph,
                                  node_travel_times: dict, profile: dict) -> gpd.GeoDataFrame:
    """
    Calculates the actual walking time to each feature (POI, crosswalk, etc.).
    This function calculates time in two ways:
    1. Uses Dijkstra travel time on the street network to the nearest node.
    2. Calculates time from nearest node to the feature using the profile's walking speed.
    """
    if gdf_features.empty:
        return gdf_features

    # Find nearest node for each feature
    nearest_nodes = ox.distance.nearest_nodes(
        graph, gdf_features.geometry.x, gdf_features.geometry.y
    )
    gdf_features['nearest_node'] = nearest_nodes
    gdf_features['time_to_node'] = gdf_features['nearest_node'].map(node_travel_times)

    # Calculate straight-line distance from node to feature
    nodes_geom = {
        node: Point(data['x'], data['y'])
        for node, data in graph.nodes(data=True)
    }
    
    # Map nearest node geometries
    gdf_features['node_geom'] = gdf_features['nearest_node'].map(nodes_geom)
    
    # Calculate distances using vectorized operation
    gdf_features['dist_from_node_to_feature'] = gdf_features.apply(
        lambda row: row.geometry.distance(row.node_geom) if row.node_geom else 0, axis=1
    )

    # Calculate time from node to feature using profile's walking speed
    walking_speed_kmh = profile['speed_walk']  # km/h from profile
    walking_speed_m_per_min = walking_speed_kmh * 1000 / 60  # convert to meters per minute
    gdf_features['time_from_node_to_feature'] = (
        gdf_features['dist_from_node_to_feature'] / walking_speed_m_per_min
    )

    # Calculate total time
    gdf_features['real_time'] = gdf_features['time_to_node'] + gdf_features['time_from_node_to_feature']

    # Remove intermediate columns for cleanup
    gdf_features.drop(columns=['dist_from_node_to_feature', 'time_from_node_to_feature', 'node_geom'], inplace=True)
    return gdf_features

def calculate_attractiveness_threshold(walking_time: float) -> float:
    """
    Calculates the Attractiveness Threshold based on walking time.

    This model uses a cosine curve for smooth decay between 0 and 20 minutes,
    and defines the threshold as 0 for any time equal to or greater than 20 minutes.
    """
    if walking_time < 0:
        return 0.0
    if walking_time >= 20:
        return 0.0
    threshold = (1 + math.cos(math.pi * walking_time / 20)) / 2
    return threshold

def process_features(features_dict: dict, graph, node_travel_times: dict, location: str, profile: dict, base_dir: str = 'data', distance: int = None) -> dict:
    
    processed_features = {}
    
    for feature_name, gdf in features_dict.items():
        print(f"\nCalculating real walking time for {feature_name}...")
        processed_gdf = calculate_feature_travel_time(gdf, graph, node_travel_times, profile)
        processed_features[feature_name] = processed_gdf
    
    return processed_features

def filter_accessible_features(features_dict: dict, max_time: float) -> dict:
    """
    Filters accessible features within maximum time.
    """
    accessible_features = {}
    
    for feature_name, gdf in features_dict.items():
        accessible_gdf = gdf[gdf['real_time'] <= max_time].copy()
        accessible_features[feature_name] = accessible_gdf
        
        print(f"Found {len(accessible_gdf)} {feature_name} "
              f"within {max_time} minutes walking distance.")
    
    return accessible_features

