import numpy as np

#############
# Constants #
#############

## every number in the interval (-epsilon, epsilon) is considered zero
epsilon = 1e-8


def real_eq(x: np.ndarray, y: np.ndarray, eps:float = epsilon) -> bool:
    assert x.shape == y.shape

    return bool((np.abs(x - y) <= eps * np.ones(x.shape)).all())


def real_neq(x: np.ndarray, y: np.ndarray, eps:float = epsilon) -> bool:
    assert x.shape == y.shape

    return bool((np.abs(x - y) > eps * np.ones(x.shape)).all())


def real_leq(x: np.ndarray, y: np.ndarray, eps:float = epsilon) -> bool:
    assert x.shape == y.shape

    return bool((x <= y + eps * np.ones(x.shape)).all())


def real_less(x: np.ndarray, y: np.ndarray, eps:float = epsilon) -> bool:
    assert x.shape == y.shape

    return bool((x < y + eps * np.ones(x.shape)).all())


def real_geq(x: np.ndarray, y: np.ndarray, eps:float = epsilon) -> bool:
    assert x.shape == y.shape

    return bool((x >= y + eps * np.ones(x.shape)).all())


def real_greater(x: np.ndarray, y: np.ndarray, eps:float = epsilon) -> bool:
    assert x.shape == y.shape

    return bool((x > y + eps * np.ones(x.shape)).all())