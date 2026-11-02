"""Integration checks with the supplied CSVs and real, shortened search runs.

Run from the project root: python -m unittest discover -s tests -v
Only the test configuration shortens searches; production defaults are intact.
"""

import contextlib
import functools
import io
import importlib.util
import json
import multiprocessing
import os
from dataclasses import fields
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from metaheuristics import optimizer
from metaheuristics.core import (
    allocation_items_to_candidate_matrix,
    build_final_indicator_matrix_nd,
    build_objective_state_nd,
    objective_function,
)
from metaheuristics.core.budget import POI_DIMENSION_COLUMNS
from metaheuristics.core.types import MetaheuristicContext
from metaheuristics.methods import brkga, grasp, ils, pso
from walkability import app, map_mode
from walkability.visualization import plot_walkability_heatmap


ROOT = Path(__file__).resolve().parents[1]
DATASETS = sorted(ROOT.glob('data/location/**/*_walkability_index_*.csv'))
RECORDS = []


def load_dataset(path):
    walkability = pd.read_csv(path)
    impacts = pd.read_csv(path.with_name(path.name.replace(
        '_walkability_index_', '_hex_time_matrix_')))
    return walkability, impacts


@contextlib.contextmanager
def isolated_directory(copy_inputs=False):
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='wqi-integration-') as directory:
        target = Path(directory)
        if copy_inputs:
            shutil.copytree(ROOT / 'data', target / 'data',
                            ignore=shutil.ignore_patterns('*.png', '*.pkl', '*.tif'))
            (target / 'seeds.txt').write_text('182\n57\n', encoding='utf-8')
        os.chdir(target)
        try:
            yield target
        finally:
            os.chdir(previous)


@contextlib.contextmanager
def short_searches(workers=1):
    """Keep the real runners and dataclass types, including for spawned workers."""
    limits = dict(max_evaluations=64, max_iterations=2, max_no_improve=2,
                  local_search_neighbor_sample=4, local_search_max_steps=2)
    defaults = tuple(limits.get(field.name, field.default)
                     for field in fields(ils.ILSRuntimeConfig))
    environment = {
        'ILS_SEED_PARALLEL_WORKERS': str(workers),
        'GRASP_SEED_PARALLEL_WORKERS': str(workers),
        'ILS_EXPERIMENT_MODE': '1', 'GRASP_EXPERIMENT_MODE': '1',
        'ILS_LOG_ENABLED': '0', 'GRASP_LOG_ENABLED': '0',
        'ILS_DEBUG_MODE': '0', 'GRASP_DEBUG_MODE': '0',
        'GRASP_MAX_EVALUATIONS': '64', 'GRASP_MAX_ITERATIONS': '2',
        'GRASP_MAX_NO_IMPROVE': '2', 'GRASP_LS_NEIGHBOR_SAMPLE': '4',
        'GRASP_LS_MAX_STEPS': '2',
    }
    runners = {
        'C': functools.partial(brkga.run_brkga, size=10, iterations=3, mode=0),
        'D': functools.partial(pso.run_pso, size=4, iterations=3, mode=0, k=2),
    }
    with patch.dict(os.environ, environment), \
            patch.object(ils.ILSRuntimeConfig.__init__, '__defaults__', defaults), \
            patch.dict(optimizer.METHOD_RUNNERS, runners), \
            contextlib.redirect_stdout(io.StringIO()):
        yield


def run_case(code, path):
    df, matrix = load_dataset(path)
    relative = path.relative_to(ROOT)
    return optimizer.walk_meta_opt(
        df, matrix, 100, code, [182, 57], relative.parts[3],
        location=relative.parts[2],
    )


class IntegrationTests(unittest.TestCase):
    def check_result(self, code, path, result):
        raw = result['method_result']
        self.assertEqual(raw['status'], 'ok')
        self.assertGreaterEqual(result['method_runtime_seconds'], 0)
        self.assertEqual(result['location'], path.relative_to(ROOT).parts[2])
        artifacts = raw if code in 'AB' else result['result_artifacts']
        allocation_file = Path(artifacts['global_best_allocation_file'])
        self.assertTrue(allocation_file.is_file())
        allocation = pd.read_csv(allocation_file)
        self.assertEqual(list(allocation.columns), ['h3_id', 'dimension', 'quantity'])
        quantity = int(allocation.quantity.sum())
        if code == 'D':
            self.assertLessEqual(quantity, 100)
        else:
            self.assertEqual(quantity, 100)
        self.assertTrue((allocation.quantity >= 0).all())
        quantities = pd.to_numeric(allocation.quantity).to_numpy(dtype=np.float64)
        self.assertTrue(np.allclose(quantities, np.round(quantities)))
        df, impacts = load_dataset(path)
        dimensions = [d for d in POI_DIMENSION_COLUMNS if d in df.columns]
        state = build_objective_state_nd(df, impacts, dimensions)
        candidate = allocation_items_to_candidate_matrix(allocation.to_dict('records'), state)
        recomputed = objective_function(final_indicator_matrix=build_final_indicator_matrix_nd(
            candidate_matrix=candidate, objective_state=state))['objective_value']
        self.assertTrue(np.isfinite(recomputed))
        if code in 'AB':
            self.assertAlmostEqual(recomputed, raw['global_best_objective_value'], places=9)
            self.assertEqual(raw['n_runs'], 2)
        else:
            np.testing.assert_array_equal(candidate, raw['proposed_matrix'])
        found = map_mode._find_latest_allocation(
            {'A': 'ils', 'B': 'grasp', 'C': 'brkga', 'D': 'pso'}[code],
            result['location'], result['profile_key'])
        self.assertEqual(Path(found).resolve(), allocation_file.resolve())
        after = map_mode._recompute_iqc_with_allocation(df, impacts, allocation)
        self.assertEqual(len(after), len(df))
        self.assertTrue(np.isfinite(after.IQC).all())
        points = map_mode._installed_pois_gdf(allocation, 'EPSG:3857')
        self.assertEqual(int(points.quantity.sum()), quantity)
        self.assertIsNotNone(app._extract_method_runtime_seconds(result))
        return {
            'method': code, 'dataset': str(path.relative_to(ROOT)),
            'allocated_quantity': quantity, 'matrix_objective_value': recomputed,
            'returned_objective_value': float(raw.get('global_best_objective_value',
                                                     raw.get('objective_value', 0))),
            'status': 'passed',
        }

    def test_all_four_methods_on_all_supplied_datasets(self):
        self.assertEqual(len(DATASETS), 15)
        with isolated_directory(), short_searches():
            for path in DATASETS:
                for code in 'ABCD':
                    with self.subTest(dataset=path.name, method=code):
                        result = run_case(code, path)
                        RECORDS.append(self.check_result(code, path, result))

    def test_legacy_context_positional_constructor(self):
        df, impacts = load_dataset(DATASETS[0])
        values = [df, impacts, 100, 'C', 'BRKGA', [182], 'athlete',
                  POI_DIMENSION_COLUMNS, df.h3_id.tolist(), 1.0, [], None]
        context = MetaheuristicContext(*values)
        self.assertEqual(context.location, 'unknown_location')
        self.assertIsNone(context.objective_state_nd)

    def test_legacy_optimizer_call_without_location(self):
        df, matrix = load_dataset(DATASETS[0])
        with isolated_directory(), short_searches():
            result = optimizer.walk_meta_opt(df, matrix, 100, 'C', [182, 57], 'athlete')
            self.assertEqual(result['location'], 'unknown_location')
            self.assertEqual(result['method_result']['status'], 'ok')

    def test_multi_profile_helper(self):
        paths = [p for p in DATASETS if p.relative_to(ROOT).parts[2] == 'av_paulista']
        payload = {}
        for path in paths:
            df, impacts = load_dataset(path)
            payload[path.relative_to(ROOT).parts[3]] = {
                'df_walkability': df, 'df_hex_time_matrix': impacts}
        with isolated_directory(), short_searches():
            result = optimizer.walk_meta_opt_multi_profile(
                payload, 100, 'C', [182, 57], location='av_paulista')
            self.assertEqual(len(result['profile_results']), 3)
            self.assertTrue(all(r['method_result']['status'] == 'ok'
                                for r in result['profile_results'].values()))

    def test_cli_mode_two_dispatches_each_method(self):
        for code in 'ABCD':
            with self.subTest(method=code), isolated_directory(copy_inputs=True), short_searches():
                results = []
                def capture(*args, **kwargs):
                    value = optimizer.walk_meta_opt(*args, **kwargs)
                    results.append(value)
                    return value
                with patch('builtins.input', side_effect=['2', '1', '1', '1', code]), \
                        patch.object(app, 'walk_meta_opt', side_effect=capture):
                    app.run_cli()
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]['method_result']['status'], 'ok')

    def test_cli_mode_three_runs_all_locations_and_profiles(self):
        with isolated_directory(copy_inputs=True), short_searches():
            results = []
            def capture(*args, **kwargs):
                value = optimizer.walk_meta_opt(*args, **kwargs)
                results.append(value)
                return value
            with patch('builtins.input', side_effect=['3', 'C']), \
                    patch.object(app, 'walk_meta_opt', side_effect=capture):
                app.run_cli()
            self.assertEqual(len(results), 15)
            self.assertEqual(len({r['location'] for r in results}), 5)
            self.assertTrue(all(r['method_result']['status'] == 'ok' for r in results))

    def test_ils_and_grasp_with_spawned_seed_workers(self):
        from concurrent.futures import ProcessPoolExecutor
        executor = functools.partial(ProcessPoolExecutor,
                                     mp_context=multiprocessing.get_context('spawn'))
        for code, module in [('A', ils), ('B', grasp)]:
            with self.subTest(method=code), isolated_directory(), short_searches(workers=2), \
                    patch.object(module, 'ProcessPoolExecutor', executor):
                result = run_case(code, DATASETS[0])
                self.assertEqual(result['method_result']['status'], 'ok')
                self.assertEqual(result['method_result']['seed_parallel_workers'], 2)
                self.assertEqual(result['method_result']['n_runs'], 2)

    def test_legacy_runtime_result_fields(self):
        self.assertEqual(app._extract_method_runtime_seconds(
            {'method_result': {'total_time': 1.25}}), 1.25)
        self.assertEqual(app._extract_method_runtime_seconds(
            {'method_runtime_seconds': 4.0, 'method_result': {'total_time': 1.25}}), 4.0)

    def test_map_mode_loads_each_methods_allocation(self):
        graph = nx.MultiDiGraph(crs='EPSG:4326')
        empty = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
        selected_locations = [((-23.5659512, -46.6557966), 'BK Avenida Paulista',
                               None, 'av_paulista')]
        with isolated_directory(copy_inputs=True), short_searches():
            for code, method in [('A', 'ils'), ('B', 'grasp'), ('C', 'brkga'), ('D', 'pso')]:
                with self.subTest(method=method):
                    run_case(code, DATASETS[0])
                    with patch.object(map_mode, '_ask_overlay_method', return_value=method), \
                            patch.object(map_mode, '_get_or_build_graph', return_value=graph), \
                            patch.object(map_mode, 'get_green_and_water_areas', return_value=(empty, empty)), \
                            patch.object(map_mode, '_get_colored_pois', return_value=empty), \
                            patch.object(map_mode, 'get_center_node', return_value=1), \
                            patch.object(map_mode, 'plot_walkability_heatmap') as plot:
                        map_mode.run_iqc_map_generation(selected_locations, ['athlete'], 9, 2000)
                    self.assertEqual(plot.call_count, 2)
                    after = plot.call_args_list[1].kwargs
                    self.assertEqual(after['filename_suffix'], '_after_' + method)
                    self.assertTrue(np.isfinite(after['df_walkability'].IQC).all())

    def test_heatmap_renders_with_local_graph_and_installed_pois(self):
        df, _ = load_dataset(DATASETS[0])
        df = df.head(20).copy()
        xmin, xmax = df.longitude.min() - 0.002, df.longitude.max() + 0.002
        ymin, ymax = df.latitude.min() - 0.002, df.latitude.max() + 0.002
        graph = nx.MultiDiGraph(crs='EPSG:4326')
        coordinates = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        for index, (x, y) in enumerate(coordinates):
            graph.add_node(index, x=x, y=y)
        for index in range(4):
            nxt = (index + 1) % 4
            graph.add_edge(index, nxt, length=100,
                           geometry=LineString([coordinates[index], coordinates[nxt]]))
        empty = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
        allocation = pd.DataFrame({'h3_id': [df.h3_id.iloc[0]],
                                   'dimension': ['S_saude'], 'quantity': [2]})
        points = map_mode._installed_pois_gdf(allocation, 'EPSG:4326')
        with isolated_directory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            plot_walkability_heatmap(df, graph, 0, empty, empty, 'local_test', 'athlete',
                                     base_dir=str(temporary), distance=2000,
                                     title='Integration validation',
                                     gdf_installed_pois=points, filename_suffix='_after_test')
            images = list((temporary / 'visualizations').glob('*.png'))
            expected = 4 if importlib.util.find_spec('mapclassify') else 1
            self.assertEqual(len(images), expected)
            self.assertTrue(all(p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for p in images))


if __name__ == '__main__':
    unittest.main()
