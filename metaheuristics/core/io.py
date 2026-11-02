import os
from typing import List


def load_seeds(seed_file_path: str = 'seeds.txt') -> List[int]:
    """Load integer seeds from file."""
    if not os.path.exists(seed_file_path):
        raise FileNotFoundError(f"Seeds file not found: {seed_file_path}")

    seeds = []
    with open(seed_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if not token:
                continue
            seeds.append(int(token))

    if not seeds:
        raise ValueError("Seeds file is empty.")

    return seeds

