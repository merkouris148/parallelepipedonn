import numpy as np

#########
# Norms #
#########
def inf_norm(x: np.ndarray) -> float:
    return np.max(np.abs(x))