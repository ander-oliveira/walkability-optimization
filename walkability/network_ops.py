from .common import *

def get_center_node(graph, central_point: tuple):
    
    central_point_gdf = gpd.GeoDataFrame(
        [1], geometry=[Point(central_point[1], central_point[0])], 
        crs="EPSG:4326"
    ).to_crs(graph.graph['crs'])
    
    central_point_projected = central_point_gdf.geometry.iloc[0]
    # start_node = ox.nearest_nodes(graph, start_lon, start_lat)
    return ox.distance.nearest_nodes(graph, central_point_projected.x, central_point_projected.y)

def compute_edge_tobler(graph: nx.MultiGraph, profile: dict, profile_key: Optional[str] = None) -> nx.MultiGraph:
    """
    Calculates crossing time using a specific pedestrian profile.
    """
    speed_walk = profile['speed_walk']
    uphill_factor = profile['uphill_factor']
    downhill_factor = profile['downhill_factor']

    edge_times = {}
    elevations = nx.get_node_attributes(graph, 'elevation')
    
    for u, v, k, data in graph.edges(data=True, keys=True):
        length = data.get('length', 0)
        if length == 0:
            edge_times[(u, v, k)] = 0
            continue

        if u in elevations and v in elevations:
            slope = (elevations[v] - elevations[u]) / length
            
            base_v_kmh = speed_walk * math.exp(-3.5 * abs(slope + 0.05))

            if slope > 0:
                v_kmh = base_v_kmh * uphill_factor
            elif slope < 0:
                v_kmh = base_v_kmh * downhill_factor
            else:
                v_kmh = base_v_kmh
        else:
            v_kmh = speed_walk
            
        v_ms = v_kmh / 3.6
        time_min = (length / v_ms) / 60 if v_ms > 0 else float('inf')
        edge_times[(u, v, k)] = time_min

    nx.set_edge_attributes(graph, edge_times, 'time')
    graph.graph['_time_profile_key'] = profile_key or profile.get('name', 'unknown_profile')
    return graph

def compute_node_travel_times(graph, center_node: int) -> dict:
    """
    Calculates minimum times to reach each node from the central node.
    """
    return nx.single_source_dijkstra_path_length(graph, center_node, weight="time")

def assign_edge_colors(graph, node_travel_times: dict, iso_intervals: list, 
                      iso_colors: list) -> None:
    
    for u, v, k, data in graph.edges(data=True, keys=True):
        t_u = node_travel_times.get(u, float("inf"))
        t_v = node_travel_times.get(v, float("inf"))
        edge_time = min(t_u, t_v)
        data["travel_time"] = edge_time
        
        if edge_time <= iso_intervals[0]: 
            data["edge_color"], data["edge_lw"] = iso_colors[0], 2
        elif edge_time <= iso_intervals[1]: 
            data["edge_color"], data["edge_lw"] = iso_colors[1], 2
        elif edge_time <= iso_intervals[2]: 
            data["edge_color"], data["edge_lw"] = iso_colors[2], 2
        elif edge_time <= iso_intervals[3]: 
            data["edge_color"], data["edge_lw"] = iso_colors[3], 2
        else: 
            data["edge_color"], data["edge_lw"] = "black", 0.5

