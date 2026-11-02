"""Export matrix results without changing the PSO/BRKGA runners or their returns."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .core.types import MetaheuristicContext
from .core.evaluation import build_final_indicator_matrix_nd, objective_function


def _slugify(value: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))
    return safe.strip('_').lower() or 'unknown'


def export_matrix_result(context: MetaheuristicContext,
                         method_result: dict,
                         method_runtime_seconds: float) -> Dict[str, str]:
    """Write map-compatible allocations alongside the original runners' outputs.

    ILS/GRASP already export allocations. PSO/BRKGA return ``proposed_matrix``;
    their original result dictionaries and legacy files are left untouched.
    """
    method_dir = {'C': 'brkga', 'D': 'pso'}.get(context.method_code)
    if method_dir is None or method_result.get('status') != 'ok':
        return {}
    matrix = method_result.get('proposed_matrix')
    state = context.objective_state_nd
    if matrix is None or state is None:
        return {}

    candidate = np.asarray(matrix)
    expected_shape = (len(state.h3_ids), len(state.candidate_dimensions))
    if candidate.shape != expected_shape:
        raise ValueError(
            f'{method_dir} returned allocation shape {candidate.shape}; '
            f'expected {expected_shape}.'
        )

    rows, cols = np.nonzero(candidate)
    allocation = pd.DataFrame({
        'h3_id': [state.h3_ids[int(row)] for row in rows],
        'dimension': [state.candidate_dimensions[int(col)] for col in cols],
        'quantity': candidate[rows, cols].astype(np.int64),
    }).sort_values(['h3_id', 'dimension']).reset_index(drop=True)

    dataset_tag = f'dataset_hex{len(state.h3_ids)}_impacts{state.source_indices.size}'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_id = f'run_{timestamp}_budget{context.budget}_seeds{len(context.seeds)}'
    root = (Path('results') / method_dir / _slugify(context.location)
            / _slugify(context.walking_profile) / dataset_tag / run_id)
    summary_dir = root / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=False)
    allocation_file = summary_dir / 'global_best_allocation.csv'
    allocation.to_csv(allocation_file, index=False)

    matrix_objective = float(objective_function(
        final_indicator_matrix=build_final_indicator_matrix_nd(
            candidate_matrix=candidate, objective_state=state,
        ),
    )['objective_value'])
    returned_objective = float(method_result['objective_value'])

    summary = {
        'method_code': context.method_code,
        'method_name': context.method_name,
        'location': context.location,
        'walking_profile': context.walking_profile,
        'budget': int(context.budget),
        'seeds': [int(seed) for seed in context.seeds],
        'returned_objective_value': returned_objective,
        'matrix_objective_value': matrix_objective,
        'objective_matches_matrix': bool(np.isclose(returned_objective, matrix_objective,
                                                    rtol=0.0, atol=1e-8)),
        'method_runtime_seconds': float(method_runtime_seconds),
        'returned_total_time': float(method_result['total_time']),
        'allocation_shape': list(candidate.shape),
        'allocated_quantity': int(allocation['quantity'].sum()),
        'allocation_file': str(allocation_file.resolve()),
        'message': 'Allocation exported from the unmodified method return.',
    }
    summary_file = summary_dir / 'matrix_result.json'
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    return {
        'output_dir': str(root.resolve()),
        'global_best_allocation_file': str(allocation_file.resolve()),
        'matrix_result_file': str(summary_file.resolve()),
    }
