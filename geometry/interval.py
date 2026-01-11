###########################################################
# geometry.interval
# --------------------------------------------------------
# A library implementing an interval in IR^d
###########################################################


#############
# Libraries #
#############
# 3rd party libraries
import numpy as np

# custom libraries
from geometry.norms import inf_norm
from geometry.constants import epsilon


###########################################################
# Class: Interval
# --------------------------------------------------------
# * Implementing the interval in IR^d.
# * An interval is a pair of vectors [lb, ub].
# * The operator "in" is overloaded and will be inherited
# in each subclass.
###########################################################
class Interval:

    ## Constructor
    def __init__(self, lb, ub):
        assert lb.shape == ub.shape
        assert (lb <= ub).all()

        ## Get dimenstions
        self.row_dim    = lb.shape[0]
        self.column_dim = lb.shape[1]
        self.dim        = self.row_dim * self.column_dim

        ## Store lb, ub
        self.lb = lb
        self.ub = ub
    
    def __copy__(self):
        return Interval(self.lb.copy(), self.ub.copy())
    
    ## Local Invariant Concistency
    def inequality_consistency(self, ind):
        return self.ub[ind] >= self.lb[ind]

    ## Global Invariant Concistency 
    def inequalities_consistency(self):
        return (self.ub >= self.lb).all()
    
    
    ## Accessors
    def get_interval(self):
        return self


    ## Mutators
    def update_lb(self, ind, val):
        assert 0 <= ind[0] and ind[0] <= self.row_dim
        assert 0 <= ind[1] and ind[1] <= self.column_dim

        self.lb[ind] = val

        assert self.inequality_consistency(ind)
    
    def update_ub(self, ind, val):
        assert 0 <= ind[0] and ind[0] <= self.row_dim
        assert 0 <= ind[1] and ind[1] <= self.column_dim

        self.ub[ind] = val

        assert self.inequality_consistency(ind)
    

    ## Interval Algebra
    def __contains__(self, x):
        assert x.shape[0] == self.row_dim
        assert x.shape[1] == self.column_dim

        return (x - self.lb >= -epsilon).all() and (self.ub - x >= -epsilon).all()

    def __eq__(self, interval) -> bool:
        if interval.row_dim     != self.row_dim:       return False
        if interval.column_dim  != self.column_dim:    return False

        return (self.lb == interval.lb).all() and (self.ub == interval.ub).all()

    def includes(self, interv):
        return interv.lb in self and interv.ub in self

    ## self <-- self \cap interval
    def intersect(self, interval):
        ## Preconditions
        assert self.row_dim     == interval.row_dim
        assert self.column_dim  == interval.column_dim
        assert (interval.ub >= interval.lb).all()

        self.lb = np.maximum(self.lb, interval.lb)
        self.ub = np.minimum(self.ub, interval.ub)

        ## Postcondition
        assert self.inequalities_consistency()
    
    ## self <-- self \sqcup interval
    def concatenate(self, interval):
        ## Preconditions
        assert self.row_dim     == interval.row_dim
        assert self.column_dim  == interval.column_dim
        assert (interval.ub >= interval.lb).all()

        self.lb = np.minimum(self.lb, interval.lb)
        self.ub = np.maximum(self.ub, interval.ub)

        ## Postcondition
        assert self.inequalities_consistency()


    # Minkowski sum
    def __add__(self, interval):
        lb_new = self.lb + interval.lb
        ub_new = self.ub + interval.ub

        new_interval = Interval(lb_new, ub_new)
        return new_interval
        
    def __sub__(self, interval):
        lb_new = self.lb - interval.lb
        ub_new = self.ub - interval.ub

        new_interval = Interval(lb_new, ub_new)
        return new_interval

    ## Metrics
    # Average coordinate-wise diameter as potential
    def calc_potential(self):
        return round(np.sum(self.ub - self.lb) / self.dim, 4)

    def get_diameter(self):
        return inf_norm(self.ub - self.lb)

    def volume(self):
        edge_lengths    = np.abs(self.ub - self.lb)
        vol             = np.prod(edge_lengths, dtype=np.int64)

        return vol
    
    def min_edge_length(self):
        return np.min(self.ub - self.lb)
    
