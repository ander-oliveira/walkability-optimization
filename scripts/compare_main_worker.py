import argparse
import contextlib
import functools
import io
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import pandas as pd
    from metaheuristics import optimizer
    from metaheuristics.methods import brkga, pso
    data_root = args.data.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix='wqi-main-compare-') as directory:
        os.chdir(directory)
        for path in sorted(data_root.glob('data/location/**/*_walkability_index_*.csv')):
            relative = path.relative_to(data_root)
            location, profile = relative.parts[2:4]
            df = pd.read_csv(path)
            impact = pd.read_csv(path.with_name(path.name.replace(
                '_walkability_index_', '_hex_time_matrix_')))
            for code, runner in [('C', brkga.run_brkga), ('D', pso.run_pso)]:
                for mode in [0, 1]:
                    options = dict(size=10 if code == 'C' else 4, iterations=3, mode=mode)
                    if code == 'D': options['k'] = 2
                    optimizer.METHOD_RUNNERS[code] = functools.partial(runner, **options)
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = optimizer.walk_meta_opt(df, impact, 100, code, [182, 57], profile)
                    raw = result['method_result']
                    key = f'{location}_{profile}_{code}_mode{mode}'
                    np.save(args.output / (key + '.npy'), raw['proposed_matrix'])
                    records.append({'key': key, 'objective_value': float(raw['objective_value']),
                                    'status': raw['status'], 'method_code': raw['method_code'],
                                    'message': raw['message']})
    (args.output / 'results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Completed',len(records),'reference cases for',args.source)


if __name__ == '__main__':
    main()
