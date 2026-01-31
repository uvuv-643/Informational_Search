import sys
import time
from typing import List

import pandas as pd


def get_duration(numbers: List[int]) -> float:
    for i in range(10):
        for j in range(3):
            print(f'stdout #{j} in sleep iteration #{i}')
            print(f'stderr #{j} in sleep iteration #{i}', file=sys.stderr)
        time.sleep(3)
    return float(pd.Series(numbers).mean())