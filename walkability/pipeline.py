from .common import *

from .data_sources import reproject_dem, get_green_and_water_areas, get_pois, get_crosswalks, get_traffic_signals
from .features import map_poi_colors_and_types, process_features, filter_accessible_features
from .hexagons import build_h3_hexagons_dataframe, get_h3_centroid
from .indicators import aggregate_hexagon_indicators
from .network_ops import get_center_node, compute_edge_tobler, compute_node_travel_times, assign_edge_colors
from .utils import ensure_data_directories
from .visualization import plot_basic_map, plot_map_with_isochrones, plot_map_with_pois_isochrones, plot_map_with_pois_isochrones_h3

def _step_1_setup_graph_env(central_point, distance, network_type, location, dem_path, base_dir, 
                           reuse_graph, reuse_green_water, generate_visualizations):
    """Stage 1: Graph, DEM, and Green/Water setup."""
    
    if reuse_graph is not None:
        print("-> Reusing existing graph and data from initial analysis")
        G = reuse_graph
        gdf_green, gdf_water = reuse_green_water
        center_node = get_center_node(G, central_point)
    else:
        # Configure OSMnx
        useful_tags = ox.settings.useful_tags_way.copy()
        for tag in ['sidewalk:width', 'sidewalk:left:width', 'sidewalk:right:width', 'sidewalk']:
            if tag not in useful_tags: useful_tags.append(tag)
        ox.settings.useful_tags_way = useful_tags
        
        print("Obtaining walkable street network graph...")
        G = ox.graph_from_point(central_point, dist=distance, network_type=network_type)
        print("Graph obtained successfully!")
        
        G = ox.projection.project_graph(G)
        
        # DEM Processing
        dem_path = reproject_dem(dem_path, G.graph['crs'])
        
        # Try multiprocessing first, fallback to single-core if it fails
        try:
            ox.elevation.add_node_elevations_raster(G, filepath=dem_path)
        except (TypeError, ValueError) as e:
            print(f"Warning: Multiprocessing failed ({e}), retrying with single CPU...")
            ox.elevation.add_node_elevations_raster(G, filepath=dem_path, cpus=1)

        center_node = get_center_node(G, central_point)
        
        print("Obtaining green and water areas...")
        gdf_green, gdf_water = get_green_and_water_areas(central_point, distance, G.graph['crs'])
        
        # Basic map visualization (only for initial analysis)
        if generate_visualizations:
            plot_basic_map(G, center_node, gdf_green, gdf_water,
                           "Basic Map - " + location, 
                           f"{base_dir}/visualizations/basic_map_{location}_dist{distance}")
        
    return G, center_node, gdf_green, gdf_water

def _step_2_network_analysis(G, center_node, profile, profile_key, location, iso_intervals, iso_colors,
                             base_dir, reuse_graph, force_recompute_tobler=False, distance=None):
    """Stage 2: Tobler computation and travel times."""

    has_edge_time = False
    for _, _, _, edge_data in G.edges(data=True, keys=True):
        has_edge_time = 'time' in edge_data
        break

    current_time_profile = G.graph.get('_time_profile_key')
    should_recompute = (
        force_recompute_tobler
        or (not has_edge_time)
        or (current_time_profile != profile_key)
    )

    if should_recompute:
        print(f"\n--- Simulating with profile: {profile['name']} ---")
        G = compute_edge_tobler(G, profile=profile, profile_key=profile_key)
    
    # Always run these steps to keep node_travel_times updated
    node_travel_times = compute_node_travel_times(G, center_node)
    assign_edge_colors(G, node_travel_times, iso_intervals, iso_colors)
    
    return G, node_travel_times

def _step_3_feature_processing(central_point, distance, G, node_travel_times, poi_colors, 
                              location, profile, iso_intervals, base_dir, reuse_raw_features):
    """Stage 3: POI/feature retrieval and processing."""
    
    if reuse_raw_features is not None:
        print("\n-> Reusing features from initial analysis")
        raw_features = reuse_raw_features
    else:
        print("\nObtaining features...")
        raw_features = {
            'pois': map_poi_colors_and_types(get_pois(central_point, distance, G.graph['crs']), poi_colors),
            'crosswalks': get_crosswalks(central_point, distance, G.graph['crs']),
            'traffic_signals': get_traffic_signals(central_point, distance, G.graph['crs'])
        }
    
    processed_features = process_features(raw_features, G, node_travel_times, location, profile, base_dir, distance)
    
    max_time = iso_intervals[-1]
    accessible_features = filter_accessible_features(processed_features, max_time)
    
    return raw_features, accessible_features, processed_features

def _step_4_outputs_viz(G, center_node, gdf_green, gdf_water, accessible_features, 
                       node_travel_times, central_point, distance, location, profile_key,
                       iso_colors, iso_intervals, h3_resolution, base_dir, 
                       generate_visualizations, generate_h3, force_serial_h3,
                       processed_features=None):
    """Stage 4: Reports, maps, and H3 grid outputs."""

    # 4.1 Visualizations
    if generate_visualizations:
        # Isochrones
        plot_map_with_isochrones(G, center_node, iso_colors, iso_intervals, gdf_green, 
                                gdf_water, central_point, distance,
                                title="Map with Isochrones - " + location, 
                                filename=f"{base_dir}/visualizations/map_isochrones_{location}_{profile_key}_res{h3_resolution}_dist{distance}")
        
        # POIs + Isochrones
        plot_map_with_pois_isochrones(G, center_node, iso_colors, iso_intervals, gdf_green, 
                                      gdf_water, accessible_features['pois'], 
                                      title="Map with Isochrones and POIs - " + location,
                                      filename=f"{base_dir}/visualizations/map_isochrones_pois_{location}_{profile_key}_res{h3_resolution}_dist{distance}")
        
        # H3 + POIs
        plot_map_with_pois_isochrones_h3(G, center_node, iso_colors, iso_intervals, gdf_green, 
                                         gdf_water, accessible_features['pois'], central_point, distance, 
                                         title="Map with Isochrones, POIs and H3 Grid - " + location,
                                         filename=f"{base_dir}/visualizations/map_isochrones_pois_h3_{location}_{profile_key}_res{h3_resolution}_dist{distance}",
                                         h3_resolution=h3_resolution)

    # 4.2 H3 Grid Generation
    df_hexagons = pd.DataFrame()
    if generate_h3:
        # print("\nSaving H3 hexagon information...")
        use_mp = not force_serial_h3
        df_hexagons = build_h3_hexagons_dataframe(
            G, central_point, distance, location,
            node_travel_times, h3_resolution, accessible_features,
            gdf_green, gdf_water, profile_key, base_dir,
            use_multiprocessing=use_mp,
            distance=distance,
            all_processed_features=processed_features
        )
                                             
    return df_hexagons

def run_analysis_pipeline(central_point: Tuple[float, float],
                          location: str,
                          dem_path: str,
                          profile: dict,
                          profile_key: str,
                          iso_intervals: list,
                          iso_colors: list,
                          poi_colors: dict,
                          distance: float,
                          h3_resolution: int,
                          network_type: str = "walk",
                          base_dir: str = 'data',
                          generate_h3: bool = True,
                          generate_visualizations: bool = True,
                          reuse_graph: nx.MultiDiGraph = None,
                          reuse_green_water: Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] = None,
                          reuse_raw_features: dict = None,
                          force_recompute_tobler: bool = False,
                          force_serial_h3: bool = False,
                          prepare_output_dirs: bool = True) -> dict:
    """Run the full pipeline and return analysis artifacts."""
    if prepare_output_dirs:
        ensure_data_directories(base_dir)

    G, center_node, gdf_green, gdf_water = _step_1_setup_graph_env(
        central_point, distance, network_type, location, dem_path, base_dir,
        reuse_graph, reuse_green_water, generate_visualizations
    )

    G, node_travel_times = _step_2_network_analysis(
        G, center_node, profile, profile_key, location, iso_intervals, iso_colors, base_dir,
        reuse_graph, force_recompute_tobler, distance
    )

    raw_features, accessible_features, processed_features = _step_3_feature_processing(
        central_point, distance, G, node_travel_times, poi_colors,
        location, profile, iso_intervals, base_dir, reuse_raw_features
    )

    df_hexagons_result = _step_4_outputs_viz(
        G, center_node, gdf_green, gdf_water, accessible_features,
        node_travel_times, central_point, distance, location, profile_key,
        iso_colors, iso_intervals, h3_resolution, base_dir,
        generate_visualizations, generate_h3, force_serial_h3,
        processed_features=processed_features
    )

    return {
        'graph': G,
        'center_node': center_node,
        'node_travel_times': node_travel_times,
        'gdf_green': gdf_green,
        'gdf_water': gdf_water,
        'accessible_features': accessible_features,
        'raw_features': raw_features,
        'df_hexagons': df_hexagons_result,
    }

def process_single_hexagon_optimized(args):
    """Process a single hexagon for multiprocessing."""
    (idx, hex_id, base_graph, base_green, base_water, base_features,
     dem_path, selected_profile, profile_key, ISO_INTERVALS, ISO_COLORS,
     POI_COLORS, DISTANCE, H3_RESOLUTION, NETWORK_TYPE,
     generate_hex_visualizations, analysis_base_dir) = args

    try:
        hex_centroid = get_h3_centroid(hex_id)
        # Only create per-hex output folders when hex visualizations are requested.
        hex_base_dir = (
            f"{analysis_base_dir}/hexagons/{hex_id}"
            if generate_hex_visualizations
            else analysis_base_dir
        )

        hex_results = run_analysis_pipeline(
            central_point=hex_centroid,
            location=hex_id,
            dem_path=dem_path,
            profile=selected_profile,
            profile_key=profile_key,
            iso_intervals=ISO_INTERVALS,
            iso_colors=ISO_COLORS,
            poi_colors=POI_COLORS,
            distance=DISTANCE,
            h3_resolution=H3_RESOLUTION,
            network_type=NETWORK_TYPE,
            base_dir=hex_base_dir,
            generate_h3=True,
            generate_visualizations=generate_hex_visualizations,
            reuse_graph=base_graph,
            reuse_green_water=(base_green, base_water),
            reuse_raw_features=base_features,
            force_serial_h3=True,
            prepare_output_dirs=generate_hex_visualizations,
        )

        df_res = hex_results.get('df_hexagons', pd.DataFrame())
        if df_res.empty:
            raise ValueError(f"Empty hexagon dataframe for {hex_id}")

        indicators = aggregate_hexagon_indicators(df_res)
        lat, lon = h3.cell_to_latlng(hex_id)
        indicators['h3_id'] = hex_id
        indicators['latitude'] = lat
        indicators['longitude'] = lon

        return (idx, hex_id, indicators, None)
    except Exception as e:
        return (idx, hex_id, None, str(e))

