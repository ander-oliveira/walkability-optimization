from datetime import datetime
from pathlib import Path
import re
from typing import Dict

import numpy as np

from ..core.types import MetaheuristicContext


DEBUG_ND_OUTPUT_DIR = Path("metaheuristics") / "debug_nd_arrays"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return slug or "unknown"


def save_nd_debug_matrices(context: MetaheuristicContext,
                           seed: int,
                           candidate_matrix: np.ndarray) -> Dict[str, str]:
    """
    Save baseline and initial solution ndarray matrices to text files.

    Files are written for temporary audit/control and can be removed later.
    """
    if context.objective_state_nd is None:
        return {}

    state = context.objective_state_nd
    output_dir = DEBUG_ND_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_slug = _slugify(context.method_code)
    profile_slug = _slugify(context.walking_profile)
    base_name = f"{method_slug}_{profile_slug}_seed{int(seed)}_{timestamp}"

    baseline_matrix_file = output_dir / f"{base_name}_baseline_matrix.txt"
    initial_matrix_file = output_dir / f"{base_name}_initial_candidate_matrix.txt"

    baseline_header = (
        f"matrix=baseline_matrix\n"
        f"method_code={context.method_code}\n"
        f"method_name={context.method_name}\n"
        f"walking_profile={context.walking_profile}\n"
        f"seed={int(seed)}\n"
        f"shape={state.baseline_matrix.shape}\n"
        f"row_order=h3_ids (ObjectiveStateND)\n"
        f"column_order=indicator_columns (ObjectiveStateND)\n"
        f"indicator_columns={','.join(state.indicator_columns)}"
    )
    np.savetxt(
        baseline_matrix_file,
        state.baseline_matrix,
        fmt="%.6f",
        delimiter=" ",
        header=baseline_header,
    )

    initial_header = (
        f"matrix=initial_candidate_matrix\n"
        f"method_code={context.method_code}\n"
        f"method_name={context.method_name}\n"
        f"walking_profile={context.walking_profile}\n"
        f"seed={int(seed)}\n"
        f"shape={candidate_matrix.shape}\n"
        f"row_order=h3_ids (ObjectiveStateND)\n"
        f"column_order=candidate_dimensions (ObjectiveStateND)\n"
        f"candidate_dimensions={','.join(state.candidate_dimensions)}"
    )
    np.savetxt(
        initial_matrix_file,
        np.asarray(candidate_matrix, dtype=np.float64),
        fmt="%.6f",
        delimiter=" ",
        header=initial_header,
    )

    return {
        "debug_baseline_matrix_file": str(baseline_matrix_file.resolve()),
        "debug_initial_candidate_matrix_file": str(initial_matrix_file.resolve()),
    }
