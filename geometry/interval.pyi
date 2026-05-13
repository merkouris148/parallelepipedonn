import typing as t

# 3rd party libraries
import numpy as np


class Interval:
    """
        A class encoding a *high-dimensional* interval `[lb, ub]`, where\\
        `lb, ub \in IR^d`. An interval is defined as:
        ```
        [lb, ub] = {x \in IR^d | lb <= x <= ub}
        ```
        **Data Members:**
        * `lb: np.ndarray`, the lower bound.
        * `ub: np.ndarray`, the upper bound.

        **Notes:**
        * The class is completely compatible with NumPy's ndarrays and\\
        is able to handle general multidimensional arrays instead of\\
        1D vectors.

        **Refferences:**
        * Theory of an Interval Algebra and Its Application to Numerical Analysis --
        Teruo Sunaga (1958).
    """
    
    ## Constructors
    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:
        """
            The default constructor.
        """
        ...
    
    def __copy__(self) -> Interval:
        """
            Copy constructor.
        """
        ...

    ## Reports
    def __str__(self) -> str:
        """
            Short `str` report.
        """
        ...
    def extended_str(self) -> str:
        """
            Extentent `str` report.
        """
        ...
    def report(self) -> None:
        """
            Print report
        """
        ...
    
    ## Predicates
    def empty(self) -> bool:
        """
            Checks if the interval is empty, i.e.,
            `ub < lb`.
        """
        ...
    def sigleton(self) -> bool:
        """
            Check if the interval is a sigleton, i.e.,
            `[x, x]`.
        """
        ...
    def is_positive(self) -> bool:
        """
            Check if the interval belongs to the positive
            quadrant.
        """
        ...
    def is_negative(self) -> bool:
        """
            Check if the interval belongs to the negative
            quadrant.
        """
        ...
    
    ## Interval Algebra
    # in operator
    def __contains__(self, x: np.ndarray) -> bool:
        """
            Checking if `x \in I`.
        """
        ...

    ## Equality & Inequality
    def __eq__(self, interval: object) -> bool:
        """
            Checking if `I1 == I2`. Note that two empty intervals
            are *always* empty.
        """
        ...
    def __ne__(self, interval: object) -> bool:
        """
            Checking if `I1 != I2`.
        """
        ...

    ## Comparisons
    def __lt__(self, interval: Interval) -> bool:
        """
            Checking if `I1 \subset I2`.
        """
        ...
    def __le__(self, interval: Interval) -> bool:
        """
            Checking if `I1 \subseteq I2`.
        """
        ...

    def __gt__(self, interval: Interval) -> bool:
        """
            Checking if `I1 \supset I2`.
        """
        ...
    def __ge__(self, interval: Interval) -> bool:
        """
            Checking if `I1 \supseteq I2`.
        """
        ...

    ## Operations on Intervals
    def join(self, interval: Interval) -> Interval:
        """
            Join operation on the interval latice. In math,
            ```
                I_1 <-- I_1 \/ I_2
            ```
        """
        ...
    def __or__(self, interval: Interval) -> Interval:
        """
            Join operation on the interval latice. In math,
            ```
                I = I_1 \/ I_2
            ```
            In python, we write:
            ```
                I = I_1 | I_2
            ```
        """
        ...
    def meet(self, interval: Interval) -> Interval:
        """
            Meet operation on the interval latice. In math,
            ```
                I_1 <-- I_1 /\ I_2
            ```
        """
        ...
    def __and__(self, interval: Interval) -> Interval:
        """
            Meet operation on the interval latice. In math,
            ```
                I = I_1 /\ I_2
            ```
            In python, we write:
            ```
                I = I_1 & I_2
            ```
        """
        ...
    # Minkowski sum
    def __add__(self, interval: Interval) -> Interval:
        """
            Adding 2 intervals, i.e., `I = I_1 + I_2`.
        """
        ...
    def __sub__(self, interval: Interval) -> Interval:
        """
            Subtractig 2 intervals, i.e., `I = I_1 - I_2`.
        """
        ...
    
    ## Vertices
    def get_vertices(self) -> np.ndarray:
        """
            Return the Interval's Vertices. Only for vector
            endpoints (ndim == 1) and dim <= 10
        """
        ...

    ## Metrics
    def diam(self) -> float:
        """
            Computing the *diameter* of the interval, i.e.,
            ```
                diam = ||lb - ub||_{+oo}
            ```
        """
        ...
    def vol(self) -> float:
        """
            Computing the *volume* of the interval. If `mpmath` is imported,
            we use arbitrary percision arithmetic. In not, there may be some
            serious underflow in the volume computation.
        """
        ...
    def avg_edge_len(self) -> float:
        """
            Computing the *average* edge length.
        """
        ...
    def min_edge_length(self) -> float:
        """
            Computing *minimum* edge length.
        """
        ...
    
    ## Sampling
    def random_points(self, N: int = 1000, seed:int =0) -> np.ndarray:
        """
            Returning `N` number of *uniformly* drawn points inside the
            interval. With `seed` we denote the seed of the Random Number
            Generator.
        """
        ...
    
