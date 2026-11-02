import random
from typing import Dict, Iterable, List


POI_DIMENSION_COLUMNS = [
    'S_saude',
    'S_educacao',
    'S_abastecimento',
    'S_lazer',
    'S_servicos',
    'T_transporte',
    'U_urbanidade',
]


def random_budget_allocation(budget: int, dimensions: Iterable[str], seed: int) -> Dict[str, int]:
    """Randomly distribute budget across dimensions using a deterministic seed."""
    dimensions = list(dimensions)
    if budget <= 0:
        raise ValueError("BUDGET must be greater than zero.")
    if not dimensions:
        raise ValueError("Dimensions list cannot be empty.")

    allocation = {dim: 0 for dim in dimensions}
    rng = random.Random(seed)

    for _ in range(budget):
        chosen_dim = rng.choice(dimensions)
        allocation[chosen_dim] += 1

    return allocation


def generate_random_allocations(budget: int,
                                dimensions: Iterable[str],
                                seeds: Iterable[int]) -> List[Dict[str, object]]:
    """Generate one random budget allocation per seed."""
    allocations = []
    for seed in seeds:
        allocations.append({
            'seed': int(seed),
            'allocation': random_budget_allocation(budget, dimensions, int(seed)),
        })
    return allocations


def random_spatial_budget_allocation(budget: int,
                                     dimensions: Iterable[str],
                                     source_hex_ids: Iterable[str],
                                     seed: int) -> List[Dict[str, object]]:
    """
    Randomly distribute budget across (source_hex, dimension) pairs.

    Returns a compact list of non-zero allocations:
    [{'h3_id': '...', 'dimension': 'S_saude', 'quantity': 3}, ...]
    """
    dimensions = [str(dim) for dim in dimensions]
    source_hex_ids = [str(h3_id) for h3_id in source_hex_ids]

    if budget <= 0:
        raise ValueError("BUDGET must be greater than zero.")
    if not dimensions:
        raise ValueError("Dimensions list cannot be empty.")
    if not source_hex_ids:
        raise ValueError("Source hexagon list cannot be empty.")

    rng = random.Random(seed)
    allocation_counter = {}

    for _ in range(budget):
        chosen_hex = rng.choice(source_hex_ids)
        chosen_dim = rng.choice(dimensions)
        key = (chosen_hex, chosen_dim)
        allocation_counter[key] = allocation_counter.get(key, 0) + 1

    allocation_items = []
    for (h3_id, dimension), quantity in allocation_counter.items():
        allocation_items.append({
            'h3_id': h3_id,
            'dimension': dimension,
            'quantity': int(quantity),
        })

    allocation_items.sort(key=lambda item: (item['h3_id'], item['dimension']))
    return allocation_items


def generate_random_spatial_allocations(budget: int,
                                        dimensions: Iterable[str],
                                        source_hex_ids: Iterable[str],
                                        seeds: Iterable[int]) -> List[Dict[str, object]]:
    """Generate one random spatial allocation per seed."""
    allocations = []
    for seed in seeds:
        allocations.append({
            'seed': int(seed),
            'allocation': random_spatial_budget_allocation(
                budget=budget,
                dimensions=dimensions,
                source_hex_ids=source_hex_ids,
                seed=int(seed),
            ),
        })
    return allocations
