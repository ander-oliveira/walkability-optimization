from .common import *

from .features import calculate_attractiveness_threshold

def h3_to_polygon_cached(h3_index: str, target_crs_str: str = None) -> str:
    """
    Cached version that returns WKT string for serialization.
    """
    coords = h3.cell_to_boundary(h3_index)
    polygon = Polygon([(lon, lat) for lat, lon in coords])
    
    if target_crs_str:
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs='EPSG:4326')
        polygon = gdf.to_crs(target_crs_str).geometry[0]
    return polygon.wkt

def h3_to_polygon(h3_index: str, target_crs: str = None) -> Polygon:
    """
    Converts an H3 index to a Shapely polygon.
    
    Returns:
        Polygon: Shapely polygon representing the H3 hexagon.
    """
    wkt_str = h3_to_polygon_cached(h3_index, target_crs)
    return wkt.loads(wkt_str)

def _process_hex_polygon(args):
    """
    Wrapper for multiprocessing hexagon polygon generation for visualization.
    Returns (hex_id, polygon, is_selected) or None if outside boundary.
    """
    hex_id, target_crs_str, map_boundary_wkt, is_selected = args
    try:
        hex_poly = h3_to_polygon(hex_id, target_crs=target_crs_str)
        map_boundary_obj = wkt.loads(map_boundary_wkt)
        
        if hex_poly.intersects(map_boundary_obj):
            return (hex_id, hex_poly, is_selected)
    except Exception as e:
        print(f"Error processing hexagon {hex_id}: {e}")
    return None

def _process_single_hexagon(i, hex_id, hex_poly, target_crs, h3_resolution,
                           node_travel_times_or_coords, accessible_features,
                           gdf_green, gdf_water, edges_gdf,
                           pois_sindex=None, crosswalks_sindex=None, signals_sindex=None,
                           edges_sindex=None, graph=None,
                           all_pois_gdf=None, all_pois_sindex=None):
    """
    Core logic for processing a single hexagon.
    Works with both serialized (multiprocessing) and direct (serial) data.
    
    Returns hexagon data dictionary with all calculated metrics.
    """
    centroid_lat, centroid_lon = h3.cell_to_latlng(hex_id)
    centroid_gdf = gpd.GeoDataFrame(
        [1], geometry=[Point(centroid_lon, centroid_lat)], 
        crs="EPSG:4326"
    ).to_crs(target_crs)
    centroid_proj = centroid_gdf.geometry.iloc[0]
    
    # Find nearest node and travel time (handles both dict types)
    if isinstance(node_travel_times_or_coords, dict):
        # Check if it's the (x,y) -> (node_id, time) format (multiprocessing)
        first_key = next(iter(node_travel_times_or_coords))
        if isinstance(first_key, tuple):
            # Multiprocessing format: {(x, y): (node_id, travel_time)}
            nearest_coord = min(node_travel_times_or_coords.keys(), 
                              key=lambda n: ((n[0] - centroid_proj.x)**2 + (n[1] - centroid_proj.y)**2)**0.5)
            nearest_node, node_time_travel = node_travel_times_or_coords[nearest_coord]
        else:
            # Serial format: {node_id: travel_time}
            if graph:
                nearest_node = ox.distance.nearest_nodes(graph, centroid_proj.x, centroid_proj.y)
                node_time_travel = node_travel_times_or_coords.get(nearest_node, None)
            else:
                node_time_travel = None
    else:
        node_time_travel = None
    
    # Initialize counters
    count_pois_accessible = 0
    count_crosswalks_accessible = 0
    count_traffic_signals_accessible = 0
    
    poi_types_count = {
        # Food / Leisure
        'restaurant': 0, 'fast_food': 0, 'bar': 0, 'cafe': 0,
        'cinema': 0, 'theatre': 0,
        # Health
        'hospital': 0, 'clinic': 0, 'doctors': 0, 'dentist': 0,
        'pharmacy': 0, 'social_facility': 0, 'nursing_home': 0, 'veterinary': 0,
        # Education
        'school': 0, 'university': 0, 'college': 0, 'kindergarten': 0,
        'childcare': 0, 'library': 0,
        # Basic supply
        'supermarket': 0, 'convenience': 0, 'grocery': 0, 'greengrocer': 0,
        'bakery': 0,
        # Extended supply
        'butcher': 0, 'fishmonger': 0, 'health_food': 0, 'deli': 0,
        'farm': 0, 'marketplace': 0,
        # General services
        'bank': 0, 'atm': 0, 'hotel': 0, 'post_office': 0,
        'police': 0, 'laundry': 0, 'hairdresser': 0, 'government': 0,
        # Fitness / Culture
        'gym': 0, 'fitness_centre': 0, 'fitness_center': 0,
        'fitness_station': 0, 'sports_centre': 0, 'playground': 0,
        'arts_centre': 0, 'community_centre': 0, 'place_of_worship': 0,
        'museum': 0, 'gallery': 0,
        # Active mobility
        'bicycle_parking': 0, 'bicycle_rental': 0,
        # Public transport
        'bus_stop': 0, 'bus_station': 0, 'taxi': 0,
        'station': 0, 'halt': 0, 'tram_stop': 0,
        # Urban stay infrastructure
        'bench': 0, 'shelter': 0, 'drinking_water': 0, 'toilets': 0,
        # Other uncategorized
        'other_pois': 0
    }
    
    area_vegetation_m2 = 0.0
    area_water_m2 = 0.0
    percent_vegetation = 0.0
    percent_water = 0.0
    avg_attractiveness_threshold_pois = 0.0
    edge_travel_time = None

    # ── Solution B: per-dimension α sums and real_times ──────────────
    _DIM_POI_MAP = {
        'saude': {'hospital', 'clinic', 'doctors', 'dentist', 'pharmacy',
                  'social_facility', 'nursing_home', 'veterinary'},
        'educacao': {'school', 'university', 'college', 'kindergarten',
                     'childcare', 'library'},
        'abastecimento': {'supermarket', 'convenience', 'grocery',
                          'greengrocer', 'bakery', 'butcher', 'fishmonger',
                          'health_food', 'deli', 'farm', 'marketplace'},
        'lazer': {'restaurant', 'cafe', 'bar', 'cinema', 'theatre', 'gym',
                  'playground', 'sports_centre', 'fitness_station',
                  'arts_centre', 'community_centre', 'place_of_worship',
                  'museum', 'gallery'},
        'servicos': {'fast_food', 'bank', 'atm', 'hotel', 'bicycle_parking',
                     'bicycle_rental', 'fitness_centre', 'fitness_center',
                     'post_office', 'police', 'laundry', 'hairdresser',
                     'government'},
        'transporte': {'bus_stop', 'bus_station', 'station', 'halt',
                       'tram_stop', 'taxi', 'bicycle_rental'},
        'urbanidade': {'bench', 'shelter', 'drinking_water', 'toilets'},
    }
    alpha_sums = {d: 0.0 for d in _DIM_POI_MAP}
    times_lists = {d: '' for d in _DIM_POI_MAP}
    
    hex_area_total = h3.cell_area(hex_id, unit='m^2')
    
    # Process vegetation
    if gdf_green is not None and not gdf_green.empty:
        try:
            vegetation_intersection = gdf_green.geometry.intersection(hex_poly)
            valid_intersections = vegetation_intersection[
                vegetation_intersection.is_valid & (~vegetation_intersection.is_empty)
            ]
            if not valid_intersections.empty:
                area_vegetation_m2 = valid_intersections.area.sum()
                percent_vegetation = (area_vegetation_m2 / hex_area_total) * 100
        except Exception:
            pass
    
    # Process water
    if gdf_water is not None and not gdf_water.empty:
        try:
            water_intersection = gdf_water.geometry.intersection(hex_poly)
            valid_intersections = water_intersection[
                water_intersection.is_valid & (~water_intersection.is_empty)
            ]
            if not valid_intersections.empty:
                area_water_m2 = valid_intersections.area.sum()
                percent_water = (area_water_m2 / hex_area_total) * 100
        except Exception:
            pass
    
    # Process accessible features
    if accessible_features:
        # POIs
        if 'pois' in accessible_features and not accessible_features['pois'].empty:
            pois_df = accessible_features['pois']
            if 'real_time' in pois_df.columns:
                pois_df = pois_df[pois_df['real_time'] <= 20].copy()
            
            # Use spatial index if available
            if pois_sindex is not None:
                possible_matches_idx = list(pois_sindex.query(hex_poly, predicate='intersects'))
                if possible_matches_idx:
                    possible_matches = pois_df.iloc[possible_matches_idx]
                    pois_in_hex = possible_matches[possible_matches.geometry.within(hex_poly)]
                else:
                    pois_in_hex = pois_df.iloc[0:0]
            else:
                pois_in_hex = pois_df[pois_df.geometry.within(hex_poly)]
            
            count_pois_accessible = len(pois_in_hex)
            
            if count_pois_accessible > 0:
                if 'poi_type' not in pois_in_hex.columns:
                    pois_in_hex = pois_in_hex.copy()

                    def _resolve_poi_type(row):
                        for col in ['amenity', 'building', 'shop', 'leisure',
                                    'tourism', 'office', 'highway', 'railway']:
                            if col in row.index and pd.notna(row[col]):
                                return row[col]
                        return 'other_pois'

                    pois_in_hex['poi_type'] = pois_in_hex.apply(_resolve_poi_type, axis=1)
                
                # Count POIs by type - OPTIMIZED: Vectorized approach using value_counts()
                # Map fitness_center to fitness_centre for consistency
                poi_type_series = pois_in_hex['poi_type'].replace('fitness_center', 'fitness_centre').fillna('other_pois')
                poi_counts = poi_type_series.value_counts()
                
                # Update counts from value_counts result
                for poi_type, count in poi_counts.items():
                    if poi_type in poi_types_count:
                        poi_types_count[poi_type] = count
                    else:
                        poi_types_count['other_pois'] += count
                
                # Calculate average attractiveness threshold
                if 'attractiveness_threshold' not in pois_in_hex.columns:
                    pois_in_hex = pois_in_hex.copy()
                    if 'real_time' in pois_in_hex.columns:
                        pois_in_hex['attractiveness_threshold'] = pois_in_hex['real_time'].apply(calculate_attractiveness_threshold)
                    else:
                        pois_in_hex['attractiveness_threshold'] = np.nan
                
                threshold_sum = pois_in_hex['attractiveness_threshold'].dropna().sum()
                num_individual_pois = len(pois_in_hex['attractiveness_threshold'].dropna())
                
                if num_individual_pois > 0:
                    avg_attractiveness_threshold_pois = round(float(threshold_sum / num_individual_pois), 5)
                else:
                    avg_attractiveness_threshold_pois = 0.0

                # ── Solution B: per-dimension α sums and real_times ──
                _poi_types_norm = poi_type_series  # already normalized
                for dim_name, poi_set in _DIM_POI_MAP.items():
                    mask = _poi_types_norm.isin(poi_set)
                    dim_pois = pois_in_hex[mask]
                    valid_alpha = dim_pois['attractiveness_threshold'].dropna()
                    alpha_sums[dim_name] = round(float(valid_alpha.sum()), 5)
                    if 'real_time' in dim_pois.columns:
                        valid_times = dim_pois['real_time'].dropna()
                        times_lists[dim_name] = ','.join(
                            f'{t:.2f}' for t in valid_times
                        )
                    else:
                        times_lists[dim_name] = ''

        # Keep times_* based on all POIs (unfiltered) for downstream analyses.
        # alpha_sum_* remains computed from filtered (<= 20 min) POIs.
        if all_pois_gdf is not None and not all_pois_gdf.empty and 'real_time' in all_pois_gdf.columns:
            if all_pois_sindex is not None:
                ap_idx = list(all_pois_sindex.query(hex_poly, predicate='intersects'))
                if ap_idx:
                    ap_matches = all_pois_gdf.iloc[ap_idx]
                    all_pois_in_hex = ap_matches[ap_matches.geometry.within(hex_poly)]
                else:
                    all_pois_in_hex = all_pois_gdf.iloc[0:0]
            else:
                all_pois_in_hex = all_pois_gdf[all_pois_gdf.geometry.within(hex_poly)]

            if not all_pois_in_hex.empty:
                if 'poi_type' not in all_pois_in_hex.columns:
                    all_pois_in_hex = all_pois_in_hex.copy()

                    def _resolve_all_pt(row):
                        for c in ['amenity', 'building', 'shop', 'leisure',
                                  'tourism', 'office', 'highway', 'railway']:
                            if c in row.index and pd.notna(row[c]):
                                return row[c]
                        return 'other_pois'

                    all_pois_in_hex['poi_type'] = all_pois_in_hex.apply(_resolve_all_pt, axis=1)

                ap_types = all_pois_in_hex['poi_type'].replace(
                    'fitness_center', 'fitness_centre').fillna('other_pois')
                for dim_name, poi_set in _DIM_POI_MAP.items():
                    dim_times = all_pois_in_hex.loc[ap_types.isin(poi_set), 'real_time'].dropna()
                    times_lists[dim_name] = ','.join(f'{t:.2f}' for t in dim_times)

        # Crosswalks
        if 'crosswalks' in accessible_features and not accessible_features['crosswalks'].empty:
            if crosswalks_sindex is not None:
                possible_idx = list(crosswalks_sindex.query(hex_poly, predicate='intersects'))
                if possible_idx:
                    possible = accessible_features['crosswalks'].iloc[possible_idx]
                    crosswalks_in_hex = possible[possible.geometry.within(hex_poly)]
                else:
                    crosswalks_in_hex = accessible_features['crosswalks'].iloc[0:0]
            else:
                crosswalks_in_hex = accessible_features['crosswalks'][
                    accessible_features['crosswalks'].geometry.within(hex_poly)
                ]
            count_crosswalks_accessible = len(crosswalks_in_hex)
        
        # Traffic signals
        if 'traffic_signals' in accessible_features and not accessible_features['traffic_signals'].empty:
            if signals_sindex is not None:
                possible_idx = list(signals_sindex.query(hex_poly, predicate='intersects'))
                if possible_idx:
                    possible = accessible_features['traffic_signals'].iloc[possible_idx]
                    signals_in_hex = possible[possible.geometry.within(hex_poly)]
                else:
                    signals_in_hex = accessible_features['traffic_signals'].iloc[0:0]
            else:
                signals_in_hex = accessible_features['traffic_signals'][
                    accessible_features['traffic_signals'].geometry.within(hex_poly)
                ]
            count_traffic_signals_accessible = len(signals_in_hex)
    
    # Edge travel time calculation
    if edges_gdf is not None and not edges_gdf.empty:
        try:
            if edges_sindex is not None:
                possible_edges_idx = list(edges_sindex.query(hex_poly, predicate='intersects'))
                if possible_edges_idx:
                    edges_in_hex = edges_gdf.iloc[possible_edges_idx]
                else:
                    edges_in_hex = edges_gdf.iloc[0:0]
            else:
                edges_in_hex = edges_gdf[edges_gdf.geometry.intersects(hex_poly)]
            
            if not edges_in_hex.empty:
                intersec = edges_in_hex.geometry.intersection(hex_poly)
                lengths = intersec.length
                idx_dom = lengths.idxmax()
                tt = edges_in_hex.loc[idx_dom, 'travel_time'] if 'travel_time' in edges_in_hex.columns else None
                if pd.isna(tt) or tt is None:
                    tt = float(edges_in_hex['travel_time'].min())
                edge_travel_time = round(float(tt), 2)
        except Exception:
            pass
    
    return {
        'sequential_id': i,
        'h3_id': hex_id,
        'latitude': centroid_lat,
        'longitude': centroid_lon,
        'coord_x_proj': centroid_proj.x,
        'coord_y_proj': centroid_proj.y,
        'resolution': h3_resolution,
        'area_m2': h3.cell_area(hex_id, unit='m^2'),
        'area_vegetation_m2': round(area_vegetation_m2, 2),
        'area_water_m2': round(area_water_m2, 2),
        'percent_vegetation': round(percent_vegetation, 2),
        'percent_water': round(percent_water, 2),
        'count_pois_accessible': count_pois_accessible,
        'count_crosswalks_accessible': count_crosswalks_accessible,
        'count_traffic_signals_accessible': count_traffic_signals_accessible,
        'total_accessible_features': count_pois_accessible + count_crosswalks_accessible + count_traffic_signals_accessible,
        'pois_restaurant': poi_types_count['restaurant'],
        'pois_fast_food': poi_types_count['fast_food'],
        'pois_bar': poi_types_count['bar'],
        'pois_cafe': poi_types_count['cafe'],
        'pois_hospital': poi_types_count['hospital'],
        'pois_clinic': poi_types_count['clinic'],
        'pois_school': poi_types_count['school'],
        'pois_university': poi_types_count['university'],
        'pois_college': poi_types_count['college'],
        'pois_kindergarten': poi_types_count['kindergarten'],
        'pois_bank': poi_types_count['bank'],
        'pois_pharmacy': poi_types_count['pharmacy'],
        'pois_doctors': poi_types_count['doctors'],
        'pois_dentist': poi_types_count['dentist'],
        'pois_cinema': poi_types_count['cinema'],
        'pois_theatre': poi_types_count['theatre'],
        'pois_hotel': poi_types_count['hotel'],
        'pois_library': poi_types_count['library'],
        'pois_bicycle_parking': poi_types_count['bicycle_parking'],
        'pois_gym': poi_types_count['gym'],
        'pois_fitness_centre': poi_types_count['fitness_centre'],
        'pois_supermarket': poi_types_count['supermarket'],
        'pois_convenience': poi_types_count['convenience'],
        'pois_grocery': poi_types_count['grocery'],
        'pois_greengrocer': poi_types_count['greengrocer'],
        'pois_bakery': poi_types_count['bakery'],
        'pois_other': poi_types_count['other_pois'],
        'avg_attractiveness_threshold_pois': avg_attractiveness_threshold_pois,
        'node_time_travel': round(node_time_travel, 2) if node_time_travel is not None else None,
        'edge_travel_time': edge_travel_time,
        # ── Solution B: per-dimension α sums (pre-computed with t_max=20) ──
        'alpha_sum_saude': alpha_sums.get('saude', 0.0),
        'alpha_sum_educacao': alpha_sums.get('educacao', 0.0),
        'alpha_sum_abastecimento': alpha_sums.get('abastecimento', 0.0),
        'alpha_sum_lazer': alpha_sums.get('lazer', 0.0),
        'alpha_sum_servicos': alpha_sums.get('servicos', 0.0),
        'alpha_sum_transporte': alpha_sums.get('transporte', 0.0),
        'alpha_sum_urbanidade': alpha_sums.get('urbanidade', 0.0),
        # Real times per dimension
        'times_saude': times_lists.get('saude', ''),
        'times_educacao': times_lists.get('educacao', ''),
        'times_abastecimento': times_lists.get('abastecimento', ''),
        'times_lazer': times_lists.get('lazer', ''),
        'times_servicos': times_lists.get('servicos', ''),
        'times_transporte': times_lists.get('transporte', ''),
        'times_urbanidade': times_lists.get('urbanidade', ''),
    }

def _process_hexagon_wrapper(args):
    """
    Wrapper for multiprocessing hexagon processing.
    Handles deserialization and delegates to core processing logic.
    Returns hexagon data dictionary or None if outside boundary.
    """
    (i, hex_id, map_boundary_wkt, target_crs, node_coords_times,
     accessible_features_dict, gdf_green_dict, gdf_water_dict, 
     edges_gdf_dict, h3_resolution, all_pois_dict) = args
    
    try:
        # Deserialize boundary and hexagon
        map_boundary = wkt.loads(map_boundary_wkt)
        hex_poly = h3_to_polygon(hex_id, target_crs=target_crs)
        
        if not hex_poly.intersects(map_boundary):
            return None
        
        # Deserialize GeoDataFrames
        gdf_green = gpd.GeoDataFrame.from_dict(gdf_green_dict) if gdf_green_dict else None
        gdf_water = gpd.GeoDataFrame.from_dict(gdf_water_dict) if gdf_water_dict else None
        edges_gdf = gpd.GeoDataFrame.from_dict(edges_gdf_dict) if edges_gdf_dict else None
        
        # Deserialize accessible features
        accessible_features = {}
        if accessible_features_dict:
            for key, feature_data in accessible_features_dict.items():
                if feature_data:
                    accessible_features[key] = gpd.GeoDataFrame.from_dict(feature_data)
        
        # Deserialize all processed POIs for times_* columns
        all_pois_gdf = gpd.GeoDataFrame.from_dict(all_pois_dict) if all_pois_dict else None

        # Use unified processing function
        return _process_single_hexagon(
            i, hex_id, hex_poly, target_crs, h3_resolution,
            node_coords_times, accessible_features,
            gdf_green, gdf_water, edges_gdf,
            pois_sindex=None, crosswalks_sindex=None, signals_sindex=None,
            edges_sindex=None, graph=None,
            all_pois_gdf=all_pois_gdf, all_pois_sindex=None
        )
    
    except Exception as e:
        print(f"Error processing hexagon {hex_id}: {str(e)}")
        return None

def plot_h3_grid(ax, graph, center_lat_lon, radius_m, h3_resolution: int) -> int:
    
    target_crs = graph.graph['crs']
    
    # Create geographic boundary from graph nodes convex hull
    gdf_nodes = ox.graph_to_gdfs(graph, edges=False)
    map_boundary = gdf_nodes.union_all().convex_hull
    
    # Calculate hexagon IDs that cover the analysis area
    hex_center = h3.latlng_to_cell(center_lat_lon[0], center_lat_lon[1], h3_resolution)
    hex_radius = math.ceil(radius_m / h3.average_hexagon_edge_length(h3_resolution, unit='m'))
    hex_ids = h3.grid_disk(hex_center, hex_radius)
    
    # Plot each hexagon that intersects the map boundary
    plotted_count = 0
    for hex_id in hex_ids:
        hex_poly = h3_to_polygon(hex_id, target_crs=target_crs)
        
        if hex_poly.intersects(map_boundary):
            patch = plt.Polygon(
                list(hex_poly.exterior.coords), 
                edgecolor='#4d4d4d', 
                facecolor='none',
                linewidth=0.9,
                alpha=0.95,
                zorder=4
            )
            ax.add_patch(patch)
            plotted_count += 1

    # Fallback: if intersection filter removed everything (usually CRS/boundary mismatch),
    # still render the grid so the layer is visible in the output map.
    if plotted_count == 0:
        for hex_id in hex_ids:
            hex_poly = h3_to_polygon(hex_id, target_crs=target_crs)
            patch = plt.Polygon(
                list(hex_poly.exterior.coords),
                edgecolor='#4d4d4d',
                facecolor='none',
                linewidth=0.9,
                alpha=0.95,
                zorder=4
            )
            ax.add_patch(patch)
            plotted_count += 1

    return plotted_count

def build_h3_hexagons_dataframe(graph, center_lat_lon: tuple, radius_m: float, 
                                location: str, 
                                node_travel_times: dict,
                                h3_resolution: int,
                                accessible_features: dict = None,
                                gdf_green: gpd.GeoDataFrame = None,
                                gdf_water: gpd.GeoDataFrame = None,
                                profile_key: str = None,
                                base_dir: str = 'data',
                                use_multiprocessing: bool = True,
                                distance: int = None,
                                all_processed_features: dict = None) -> pd.DataFrame:
   
    gdf_nodes = ox.graph_to_gdfs(graph, edges=False)
    map_boundary = gdf_nodes.union_all().convex_hull
    target_crs = graph.graph['crs']

    # Prepare edges with travel_time to check hexagon intersections
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    if not edges_gdf.empty:
        # keep only what we need
        keep_cols = ['travel_time', 'geometry']
        edges_gdf = edges_gdf[[c for c in keep_cols if c in edges_gdf.columns]].copy()
        # Create spatial index for faster edge intersection queries
        if len(edges_gdf) > 0:
            edges_sindex = edges_gdf.sindex

    # Calculate hexagon IDs that cover the analysis area
    hex_center = h3.latlng_to_cell(center_lat_lon[0], center_lat_lon[1], h3_resolution)
    hex_radius = math.ceil(radius_m / h3.average_hexagon_edge_length(h3_resolution, unit='m'))
    hex_ids = list(h3.grid_disk(hex_center, hex_radius))
    
    # Create spatial indexes for features (speeds up within/intersection queries)
    if accessible_features:
        if 'pois' in accessible_features and not accessible_features['pois'].empty:
            pois_sindex = accessible_features['pois'].sindex
        if 'crosswalks' in accessible_features and not accessible_features['crosswalks'].empty:
            crosswalks_sindex = accessible_features['crosswalks'].sindex
        if 'traffic_signals' in accessible_features and not accessible_features['traffic_signals'].empty:
            signals_sindex = accessible_features['traffic_signals'].sindex
    
    # Spatial index for all processed POIs (unfiltered)
    all_pois_gdf = None
    all_pois_sindex = None
    if all_processed_features and 'pois' in all_processed_features and not all_processed_features['pois'].empty:
        all_pois_gdf = all_processed_features['pois']
        all_pois_sindex = all_pois_gdf.sindex

    total_hexagons = len(hex_ids)
    
    # MULTIPROCESSING IMPLEMENTATION
    if use_multiprocessing and total_hexagons > 100:
        
        # Serialize data for worker processes
        map_boundary_wkt = map_boundary.wkt
        
        # Prepare node_travel_times as dict with (x,y) keys for faster lookup
        node_coords_times = {}
        for node_id, travel_time in node_travel_times.items():
            node_data = graph.nodes[node_id]
            node_coords_times[(node_data['x'], node_data['y'])] = (node_id, travel_time)
        
        # Serialize GeoDataFrames to dict (picklable)
        gdf_green_dict = gdf_green.to_dict() if gdf_green is not None and not gdf_green.empty else None
        gdf_water_dict = gdf_water.to_dict() if gdf_water is not None and not gdf_water.empty else None
        edges_gdf_dict = edges_gdf.to_dict() if not edges_gdf.empty else None
        
        # Serialize accessible features
        accessible_features_dict = {}
        if accessible_features:
            for key, gdf in accessible_features.items():
                if gdf is not None and not gdf.empty:
                    accessible_features_dict[key] = gdf.to_dict()
        
        # Serialize all processed POIs for times_* columns
        all_pois_dict = all_pois_gdf.to_dict() if all_pois_gdf is not None and not all_pois_gdf.empty else None

        # Create argument tuples for each hexagon
        args_list = [
            (i+1, hex_id, map_boundary_wkt, target_crs, node_coords_times,
             accessible_features_dict, gdf_green_dict, gdf_water_dict,
             edges_gdf_dict, h3_resolution, all_pois_dict)
            for i, hex_id in enumerate(hex_ids)
        ]
        
        # Process hexagons in parallel
        num_processes = max(1, cpu_count() - 1)  # Leave 1 core free
        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_hexagon_wrapper, args_list)
        
        # Filter out None results (hexagons outside boundary)
        hexagon_data = [r for r in results if r is not None]
        
    else:
        # SERIAL PROCESSING (fallback for small datasets or disabled multiprocessing)
        hexagon_data = []
        
        # Process each hexagon that intersects the map boundary
        for i, hex_id in enumerate(hex_ids, 1):
            
            hex_poly = h3_to_polygon(hex_id, target_crs=target_crs)
            
            if hex_poly.intersects(map_boundary):
                # Use unified processing function
                result = _process_single_hexagon(
                    i, hex_id, hex_poly, target_crs, h3_resolution,
                    node_travel_times, accessible_features,
                    gdf_green, gdf_water, edges_gdf,
                    pois_sindex=pois_sindex if 'pois_sindex' in locals() else None,
                    crosswalks_sindex=crosswalks_sindex if 'crosswalks_sindex' in locals() else None,
                    signals_sindex=signals_sindex if 'signals_sindex' in locals() else None,
                    edges_sindex=edges_sindex if 'edges_sindex' in locals() else None,
                    graph=graph,
                    all_pois_gdf=all_pois_gdf,
                    all_pois_sindex=all_pois_sindex
                )
                if result:
                    hexagon_data.append(result)
    
    if hexagon_data:
        df_hexagons = pd.DataFrame(hexagon_data)
        
        # Calculate minimum access time (prioritizes best available access)
        df_hexagons['min_time_access'] = df_hexagons.apply(lambda row: 
            min(
                row['node_time_travel'] if pd.notna(row['node_time_travel']) else float('inf'),
                row['edge_travel_time'] if pd.notna(row['edge_travel_time']) else float('inf')
            ), axis=1
        )
        
        df_hexagons['min_time_access'] = df_hexagons['min_time_access'].replace([float('inf')], None)
        
        df_hexagons['min_time_access'] = df_hexagons['min_time_access'].apply(
            lambda x: round(x, 2) if pd.notna(x) else None
        )

        total_hexagons_all = len(df_hexagons)
        accessible_count = int(
            ((df_hexagons['min_time_access'].notna()) & (df_hexagons['min_time_access'] <= 20)).sum()
        )

        print(f"   Hexagons accessible (≤20min): {accessible_count}")
        
        return df_hexagons
    
    return pd.DataFrame()

def select_random_hexagons(df_hexagons: pd.DataFrame) -> list:
    """
    Selects hexagons directly from an in-memory dataframe.

    Current policy keeps all available hexagons (no random sampling),
    preserving previous behavior while avoiding intermediate CSV/TXT files.

    Args:
        df_hexagons: DataFrame containing at least the 'h3_id' column.

    Returns:
        list: List of selected H3 hexagon IDs.
    """
    if df_hexagons is None or df_hexagons.empty:
        print("⚠ No hexagons available to select.")
        return []

    if 'h3_id' not in df_hexagons.columns:
        print("⚠ Hexagon dataframe has no 'h3_id' column.")
        return []

    selected_ids = df_hexagons['h3_id'].dropna().astype(str).tolist()
    print(f"\nTotal hexagons available: {len(selected_ids)}")
    print(f"-> All available hexagons selected ({len(selected_ids)}) [in-memory].")
    return selected_ids

def get_h3_centroid(h3_id: str) -> Tuple[float, float]:
    """
    Gets the centroid (latitude, longitude) of an H3 hexagon.
    
    Args:
        h3_id: H3 hexagon ID
        
    Returns:
        Tuple with (latitude, longitude)
    """
    lat, lon = h3.cell_to_latlng(h3_id)
    return (lat, lon)

