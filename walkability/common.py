import csv
import math
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple

import geopandas as gpd
import h3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues with multiprocessing
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely import wkt
from shapely.geometry import Point, Polygon
