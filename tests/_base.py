import pandas as pd
import numpy as np


def data_example(n_samples=1000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    data = {
        "fea_1": np.random.choice([0, 1], size=n_samples),
        "fea_2": np.random.choice(list(range(3)), size=n_samples),
        "fea_3": np.random.choice(list(range(4)), size=n_samples),
        "fea_4": np.random.choice(list(range(5)), size=n_samples),
        "fea_5": np.random.randint(1, 100, size=n_samples),
        "fea_6": np.random.randint(15, 80, size=n_samples),
        "fea_7": np.random.randint(30000, 100000, size=n_samples),
        "fea_8": np.random.randint(0, 50, size=n_samples),
        "fea_9": np.random.randint(0, 6000, size=n_samples),
        "target": np.random.choice([0, 1], size=n_samples)
    }
    return pd.DataFrame(data)
