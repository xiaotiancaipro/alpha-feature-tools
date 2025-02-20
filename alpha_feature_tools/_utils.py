import itertools


def feature_names_sep() -> str:
    """Return the separator used to separate features in the feature names."""
    return "_&_"


def find_unique_subsets(data: list, n: int = 2) -> list:
    """
    Generates and returns all unique combinations of n elements in data,
    ensuring that each combination is in order.

    Parameters
    ----------
    data : list
        The list of elements to generate combinations from.

    n : int, optional
        The number of elements in each combination. Default is 2.

    Returns
    -------
    list
        A list of unique combinations of n elements in data. Each combination is in order.
    """
    combinations = itertools.combinations(data, n)
    unique_combinations = {tuple(sorted(_)) for _ in combinations}
    return list(sorted(unique_combinations))
