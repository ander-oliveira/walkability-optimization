from .brkga import run_brkga
from .grasp import run_grasp
from .hybrid_grasp_vns_pr import run_hybrid_grasp_vns_pr
from .ils import run_ils
from .pso import run_pso

METHOD_RUNNERS = {
    'A': run_ils,
    'B': run_grasp,
    'C': run_brkga,
    'D': run_pso,
    'E': run_hybrid_grasp_vns_pr,
}

__all__ = [
    'METHOD_RUNNERS',
    'run_brkga',
    'run_grasp',
    'run_hybrid_grasp_vns_pr',
    'run_ils',
    'run_pso',
]

