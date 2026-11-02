"""Run the integration suite and write its detailed, reproducible report."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'VALIDACAO_INTEGRACAO.json')
    args = parser.parse_args()
    import test_integration
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_integration)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        'successful': outcome.wasSuccessful(),
        'tests_run': outcome.testsRun,
        'failures': [str(test) + '\n' + trace for test, trace in outcome.failures],
        'errors': [str(test) + '\n' + trace for test, trace in outcome.errors],
        'python': platform.python_version(),
        'platform': platform.platform(),
        'elapsed_seconds': time.perf_counter() - started,
        'dependencies': {name: importlib.metadata.version(name) for name in [
            'numpy', 'pandas', 'scipy', 'geopandas', 'osmnx', 'h3',
            'networkx', 'shapely', 'rasterio', 'matplotlib']},
        'search_configuration': {
            'budget': 100, 'seeds': [182, 57],
            'ils_grasp': {'max_iterations': 2, 'max_evaluations': 64,
                          'local_search_max_steps': 2, 'local_search_neighbor_sample': 4},
            'brkga': {'size': 10, 'iterations': 3, 'mode': 0},
            'pso': {'size': 4, 'iterations': 3, 'mode': 0, 'k': 2},
        },
        'dataset_method_checks': test_integration.RECORDS,
        'inherited_main_findings': [
            record for record in test_integration.RECORDS
            if record['method'] in 'CD' and abs(record['returned_objective_value']
                                               - record['matrix_objective_value']) > 1e-8
        ],
        'limits': [
            'Shortened searches validate integration, not convergence or scientific performance.',
            'Mode 1 with external OSM downloads and DEM files is not exercised.',
            'Map allocations, IQC recomputation, point geometries and local heatmap rendering '
            'are exercised; live OSM downloads are not.',
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    print(f'Validation report: {args.output}')
    return 0 if outcome.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
