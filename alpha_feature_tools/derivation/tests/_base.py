import pandas as pd
import numpy as np


def data_example(n_samples=1000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    data = {
        "user_id": np.arange(1, n_samples + 1),
        "occupation": np.random.choice(list(range(5)), size=n_samples),
        "education": np.random.choice(list(range(3)), size=n_samples),
        "gender": np.random.choice([0, 1], size=n_samples),
        "region": np.random.choice(list(range(4)), size=n_samples),
        "age": np.random.randint(18, 70, size=n_samples),
        "income": np.random.randint(30000, 100000, size=n_samples),
        "purchase_frequency": np.random.randint(0, 50, size=n_samples),
        "purchase_amount": np.random.randint(0, 500, size=n_samples),
        "page_views": np.random.randint(0, 200, size=n_samples),
        "days_since_last_purchase": np.random.randint(0, 365, size=n_samples),
        "target": np.random.choice([0, 1], size=n_samples)
    }
    return pd.DataFrame(data)
