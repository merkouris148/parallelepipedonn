###########################################################
# geometry.interval
# --------------------------------------------------------
# A library implementing an interval in IR^d
###########################################################
#############
# Constants #
#############
warnings = True


#############
# Libraries #
#############
import typing as t
import itertools

# 3rd party libraries
#from pprint import pformat
import numpy as np


# arbitrary percision arithmetic
import sys
mpmath_lib_name = "mpmath"
try:
    import mpmath as mpm # type: ignore
except ImportError as _:
    if warnings:
        print("Warning [geometry.intervals]: mpmath is not installed!")
        print("Arbitrary Percision Arithmetic is disabled")


#custom libraries
# from norms import inf_norm
# from constants import epsilon
# from norms import inf_norm
# from constants import epsilon


###########################################################
# Class: Interval
# --------------------------------------------------------
# * Implementing the interval in IR^d.
# * An interval is a pair of vectors [lb, ub].
# * The operator "in" is overloaded and will be inherited
# in each subclass.
###########################################################
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
    ################
    # Constructors #
    ################
    def __init__(self, lb: np.ndarray, ub: np.ndarray):
        """
            The default constructor.
        """
        assert lb.shape == ub.shape

        ## Get dimenstions
        self.shape  = lb.shape
        self.dim    = np.prod(np.array(self.shape))

        ## Store lb, ub
        self.lb = lb
        self.ub = ub
    
    def __copy__(self):
        """
            Copy constructor.
        """
        return Interval(self.lb.copy(), self.ub.copy())
    

    ##########
    # Report #
    ##########
    def __str__(self) -> str:
        """
            Short `str` report.
        """
        s = "interval[\n"
        s +=  str(self.lb) + ",\n"
        s +=  str(self.ub) + "\n"
        s += "]"

        return s

    def extended_str(self) -> str:
        """
            Extentent `str` report.
        """
        s = "\n\n" + str(self) + "\n\n"

        s += f"{'Diam.:':<17}"          + str(self.diam())              + "\n"
        s += f"{'Min. Edge Len.:':<17}" + str(self.min_edge_length())   + "\n"
        s += f"{'Avg. Edge Len.:':<17}" + str(self.avg_edge_len())      + "\n"
        s += f"{'Vol.:':<17}"            + str(self.vol())               + "\n"

        return s
    
    def report(self):
        """
            Print report
        """
        print(self.extended_str())

    ##############
    # Predicates #
    ##############
    # Bellow we use a hacky way to force the returned type to be bool.
    # Note that NumPy returns a np.bool_ type, with np.bool_ != bool.
    # This is an inherited problem from NumPy. See:
    # https://github.com/python/mypy/issues/10385
    def empty(self) -> bool:
        """
            Checks if the interval is empty, i.e.,
            `ub < lb`.
        """
        return ((self.ub < self.lb).all()) == True
    
    def sigleton(self) -> bool:
        """
            Check if the interval is a sigleton, i.e.,
            `[x, x]`.
        """
        return ((self.ub == self.lb).all()) == True

    def is_positive(self) -> bool:
        """
            Check if the interval belongs to the positive
            quadrant.
        """
        O = np.zeros(self.shape)
        return ((self.ub >= O).all() and (self.lb >= O).all()) == True
    
    def is_negative(self) -> bool:
        """
            Check if the interval belongs to the negative
            quadrant.
        """
        O = np.zeros(self.shape)
        return ((self.ub <= O).all() and (self.lb <= O).all()) == True
    
    ####################
    # Interval Algebra #
    ####################
    ## in operator
    def __contains__(self, x) -> bool:
        """
            Checking if `x \in I`.
        """
        if self.shape != x.shape: return False
        return (self.lb <= x).all() and (x <= self.ub).all()

    ## Equality & Inequality
    def __eq__(self, interval) -> bool:
        """
            Checking if `I1 == I2`. Note that two empty intervals
            are *always* equal.
        """
        ## Adhering to Liskov substitution principle
        # Something that is not interval cannot be equal to an
        # interval
        if not isinstance(interval, Interval): return False

        if self.empty() and interval.empty(): return True

        if self.shape != interval.shape: return False
        return (self.lb == interval.lb).all() and (self.ub == interval.ub).all()

    def __ne__(self, interval) -> bool:
        """
            Checking if `I1 != I2`.
        """
        return not self == interval

    ## Comparisons
    def __lt__(self, interval) -> bool:
        """
            Checking if `I1 \subset I2`.
        """
        return (interval.lb < self.lb).all() and (self.ub < interval.ub).all()

    def __le__(self, interval) -> bool:
        """
            Checking if `I1 \subseteq I2`.
        """
        return (interval.lb <= self.lb).all() and (self.ub <= interval.ub).all()

    def __gt__(self, interval) -> bool:
        """
            Checking if `I1 \supset I2`.
        """
        return (interval.lb > self.lb).all() and (self.ub > interval.ub).all()

    def __ge__(self, interval) -> bool:
        """
            Checking if `I1 \supseteq I2`.
        """
        return (interval.lb >= self.lb).all() and (self.ub >= interval.ub).all()

    ## Operations on Intervals
    ## self <-- self \sqcup interval
    def join(self, interval):
        """
            Join operation on the interval latice. In math,
            ```
            I_1 <-- I_1 \/ I_2
            ```
        """
        assert self.shape == interval.shape

        self.lb = np.minimum(self.lb, interval.lb)
        self.ub = np.maximum(self.ub, interval.ub)
    
    def __or__(self, interval):
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
        assert self.shape == interval.shape

        lb_new = np.minimum(self.lb, interval.lb)
        ub_new = np.maximum(self.ub, interval.ub)

        return Interval(lb_new.copy(), ub_new.copy())


    # self <-- self \cap interval
    def intersect(self, interval):
        return self.meet(interval)

    def meet(self, interval):
        """
            Meet operation on the interval latice. In math,
            ```
            I_1 <-- I_1 /\ I_2
            ```
        """
        assert self.shape == interval.shape

        self.lb = np.maximum(self.lb, interval.lb)
        self.ub = np.minimum(self.ub, interval.ub)

    def __and__(self, interval):
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
        assert self.shape == interval.shape

        lb_new = np.maximum(self.lb, interval.lb)
        ub_new = np.minimum(self.ub, interval.ub)

        return Interval(lb_new.copy(), ub_new.copy())


    # Minkowski sum
    def __add__(self, interval):
        """
            Adding 2 intervals, i.e., `I = I_1 + I_2`.
        """
        assert self.shape == interval.shape

        lb_new = self.lb + interval.lb
        ub_new = self.ub + interval.ub

        return Interval(lb_new.copy(), ub_new.copy())
    

    def __sub__(self, interval):
        """
            Subtractig 2 intervals, i.e., `I = I_1 - I_2`.
        """
        assert self.shape == interval.shape

        lb_new = self.lb - interval.lb
        ub_new = self.ub - interval.ub

        new_interval = Interval(lb_new, ub_new)
        return new_interval


    ## Vertices
    def get_vertices(self) -> np.ndarray:
        """
            Return the Interval's Vertices. Only for vector
            endpoints (ndim == 1) and dim <= 10
        """

        assert self.lb.ndim == 1,\
        "Intervals.get_vertices works only for intervals\
        with vector endpoints, i.e. ndim == 1."
        assert self.lb.shape[0] <= 10,\
        "Intervals.get_vertices works only for intervals\
        with low dim vector endpoints, i.e. dim <= 10."

        Vertices    = []
        dim         = self.lb.shape[0]
        for bin_vec in itertools.product([0,1], repeat=dim):
            vertex = np.zeros((dim,))
            for ind in range(dim):
                if bin_vec[ind] == 0:   vertex[ind] = self.lb[ind]
                else:                   vertex[ind] = self.ub[ind]
            
            Vertices.append(vertex)
        
        return np.array(Vertices)
            
            

    ## Metrics
    def diam(self):
        """
            Computing the *diameter* of the interval, i.e.,
            ```
            diam = ||lb - ub||_{+oo}
            ```
        """
        return inf_norm(self.ub - self.lb)


    def vol(self):
        """
            Computing the *volume* of the interval. If `mpmath` is imported,
            we use arbitrary percision arithmetic. In not, there may be some
            serious underflow in the volume computation.
        """
        edge_lengths    = np.abs(self.ub - self.lb)
        vol             = 0.0

        if mpmath_lib_name not in sys.modules:
            if warnings: 
                print("Warning [geometry.intervals]: mpmath is not installed!")
                print("Volume computation might be inacurate!")
            
            vol = np.prod(edge_lengths, dtype=np.float64)
        
        else:
            with mpm.workdps(self.dim):
                vol = mpm.fprod(edge_lengths.flatten())

        return vol
    

    # Average coordinate-wise diameter as potential
    def avg_edge_len(self):
        """
            Computing the *average* edge length.
        """
        return round(np.sum(self.ub - self.lb) / self.dim, 4)


    def min_edge_length(self):
        """
            Computing *minimum* edge length.
        """
        return np.min(self.ub - self.lb)
    
    
    ############
    # Sampling #
    ############
    def random_points(self, N=1000, seed=0):
        """
            Returning `N` number of *uniformly* drawn points inside the
            interval. With `seed` we denote the seed of the Random Number
            Generator.
        """
        assert N > 0
        rnjeasus    = np.random.default_rng(seed)
        out_shape   = tuple([N] + list(self.shape))
        return rnjeasus.uniform(self.lb, self.ub, out_shape)
    
