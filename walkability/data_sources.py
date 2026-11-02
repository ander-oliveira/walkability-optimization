from .common import *

def reproject_dem(dem_path: str, target_crs) -> str:
    """
    Checks the DEM CRS and reprojects if necessary.
    Must be in the same form as the graph.
    """
    
    if not os.path.isabs(dem_path):
        dem_path = os.path.abspath(dem_path)
    
    try:
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
        
        if dem_crs != target_crs:
            dst_crs = target_crs
            with rasterio.open(dem_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs, 
                    'transform': transform, 
                    'width': width, 
                    'height': height
                })
                dem_dir = os.path.dirname(dem_path)
                dem_filename = os.path.basename(dem_path)
                reprojected_dem_path = os.path.join(dem_dir, f'reprojected_{dem_filename}')
                
                with rasterio.open(reprojected_dem_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i), 
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform, 
                            src_crs=src.crs,
                            dst_transform=transform, 
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest
                        )
                
                print(f"DEM reprojected to graph CRS: {reprojected_dem_path}")
                return reprojected_dem_path
        else:
            print("DEM is already in the graph CRS.")
            return dem_path
    except Exception as e:
        print(f"ERROR processing DEM file '{dem_path}': {type(e).__name__}: {e}")
        return None

def get_green_and_water_areas(place: tuple, distance: float, target_crs) -> tuple:
    
    # Check cache first
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"green_water_{place[0]:.6f}_{place[1]:.6f}_{distance}_{target_crs}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading green/water areas from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    tags_green = {
        'landuse': ['allotments', 'farmland', 'meadow', 'greenfield', 'forest', 'grass', 'park', 'greenery'], 
        'leisure': ['park', 'garden', 'nature_reserve'],
        'natural': ['fell', 'tree', 'wood', 'scrub', 'grassland']
    }
    tags_water = {
        'natural': ['bay', 'beach', 'spring', 'water', 'wetland'], 
        'water': ['river', 'lake', 'canal', 'ditch', 'reservoir', 'lagoon'],
        'waterway': ['river', 'riverbank', 'stream', 'canal', 'waterfall']
    }
    
    gdf_green = ox.features_from_point(place, tags=tags_green, dist=distance).to_crs(target_crs)
    gdf_water = ox.features_from_point(place, tags=tags_water, dist=distance).to_crs(target_crs)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump((gdf_green, gdf_water), f)
    
    return gdf_green, gdf_water

def get_pois(place: tuple, distance: float, target_crs) -> gpd.GeoDataFrame:
    
    # Check cache first
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"pois_{place[0]:.6f}_{place[1]:.6f}_{distance}_{target_crs}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading POIs from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    tags_pois = {
        'amenity': [
            # --- Food ---
            'bar', 'cafe', 'fast_food', 'restaurant',
            # --- Education ---
            'college', 'kindergarten', 'library', 'school', 'university',
            'childcare',                            # Higgs et al. (2019); Ewing & Cervero (2010)
            # --- Active mobility ---
            'bicycle_parking', 'bicycle_rental',    # Ewing & Cervero (2010)
            # --- Health ---
            'clinic', 'dentist', 'doctors', 'hospital', 'pharmacy',
            'social_facility',                      # Higgs et al. (2019) ? social support services
            'nursing_home',                         # Barnett et al. (2017) ? older population
            'veterinary',                           # Frank et al. (2010) — variantes Walk Score
            # --- Culture and leisure (S_lazer) ---
            'cinema', 'theatre',
            'arts_centre',                          # Higgs et al. (2019)
            'community_centre',                     # Higgs et al. (2019); Barnett et al. (2017)
            'place_of_worship',                     # Frank et al. (2010); recurring pedestrian destination
            # --- Fitness ---
            'gym', 'fitness_centre', 'fitness_center',
            # --- General services ---
            'bank', 'atm',                          # Frank et al. (2010); Apparicio et al. (2007)
            'hotel',
            'post_office',                          # Apparicio et al. (2007); Ewing & Cervero (2010)
            'police',                               # Higgs et al. (2019) - perceived safety
            'laundry',                              # everyday services with high pedestrian frequency
            # --- Urban stay infrastructure (U_urbanidade) ---
            'bench',                                # Barnett et al. (2017) ? essential for older adults
            'shelter',                              # pedestrian comfort; staying in public space
            'drinking_water',                       # urban stay infrastructure
            'toilets',                              # public hygiene; encourages pedestrian stay
            # --- Public transport ---
            'bus_station', 'taxi',                  # Ewing & Cervero (2010); Higgs et al. (2019)
        ],
        'building': [
            'hotel', 'hospital', 'kindergarten', 'school', 'university', 'supermarket'
        ],
        'shop': [
            # --- Basic supply ---
            'supermarket', 'convenience', 'bakery', 'greengrocer', 'grocery',
            # --- Extended supply ---
            'butcher',       # Apparicio et al. (2007); Barnett et al. (2017)
            'fishmonger',    # local food access; relevant in Brazilian riverine contexts
            'health_food',   # access to healthy foods; food environment walkability
            'deli',          # neighborhood-level food access
            'farm',          # local food supply; peri-urban context
            'marketplace',   # very relevant in Brazil ? open-air market as a key pedestrian destination
            # --- Daily services ---
            'laundry',       # high-frequency pedestrian service
            'hairdresser',   # Frank et al. (2010) - Walk Score variants
        ],
        'leisure': [
            'playground',       # Frank et al. (2010); Ewing & Cervero (2010) ? family walkability
            'sports_centre',    # Higgs et al. (2019); Barnett et al. (2017)
            'fitness_station',  # outdoor gym; common in Brazilian urban street furniture
        ],
        'tourism': [
            'museum',   # Higgs et al. (2019) ? urban cultural vitality
            'gallery',  # cultural destination; captures land-use diversity
        ],
        'office': [
            'government',  # Apparicio et al. (2007) - public services with high pedestrian demand
        ],
        # --- Public transport: additional dimension ---
        'highway': ['bus_stop'],          # Ewing & Cervero (2010); Higgs et al. (2019)
        'railway': ['station', 'halt', 'tram_stop'],  # access to rail/metro system
    }
    
    gdf_pois = ox.features_from_point(
        place, tags=tags_pois, dist=distance
    ).to_crs(target_crs)
    
    # Separate points and polygons
    points = gdf_pois[gdf_pois.geometry.geom_type == 'Point'].copy()
    polygons = gdf_pois[gdf_pois.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    
    # Convert polygon centroids to points (important for universities, hospitals, etc.)
    if not polygons.empty:
        polygons['geometry'] = polygons.geometry.centroid
        gdf_pois = pd.concat([points, polygons], ignore_index=True)
    else:
        gdf_pois = points
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(gdf_pois, f)
    
    return gdf_pois

def get_crosswalks(place: tuple, distance: float, target_crs) -> gpd.GeoDataFrame:
    
    # Check cache first
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"crosswalks_{place[0]:.6f}_{place[1]:.6f}_{distance}_{target_crs}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading crosswalks from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    tags_crosswalks = {
        'highway': 'crossing',
        'footway': 'crossing'
    }
    
    gdf_crosswalks = ox.features_from_point(
        place, tags=tags_crosswalks, dist=distance
    ).to_crs(target_crs)
    gdf_crosswalks = gdf_crosswalks[
        gdf_crosswalks.geometry.geom_type == 'Point'
    ].copy()
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(gdf_crosswalks, f)
    
    return gdf_crosswalks

def get_traffic_signals(place: tuple, distance: float, target_crs) -> gpd.GeoDataFrame:

    # Check cache first
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"signals_{place[0]:.6f}_{place[1]:.6f}_{distance}_{target_crs}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print("Loading traffic signals from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    tags_signals = {
        'highway': ['traffic_signals', 'crossing', 'stop'],
        'traffic_signals': True,
        'crossing': True
    }
    
    try:
        gdf_signals = ox.features_from_point(place, tags=tags_signals, dist=distance).to_crs(target_crs)
        gdf_signals = gdf_signals[gdf_signals.geometry.geom_type == 'Point'].copy()
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(gdf_signals, f)
        
        return gdf_signals
    except Exception as e:
        print(f"Error obtaining traffic signals: {e}")
        return gpd.GeoDataFrame()

