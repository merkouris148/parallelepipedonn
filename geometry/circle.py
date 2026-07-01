###########################################################
# geometry.circle
# --------------------------------------------------------
# A library that implements circles for different norms.
# These circles can be used both as distance restrictions
# and notions of explanation.
###########################################################


#############
# Libraries #
#############
# 3rd party libraries
import numpy as np

# typing
from typing import Callable

# Custom libraries
import geometry.interval as interval
import geometry.norms as norms
from geometry.constants import epsilon



###########
# Circles #
###########

###########################################################
# Class: Circle
# --------------------------------------------------------
# * The base class for circles.
# * Each subclass should be defined by giving an alternate
# norm.
# * The "in" operator is implemented once and works for
# each subclass
###########################################################
class Circle:
    """
        * The base class for circles.
        * Each subclass should be defined by giving an alternate norm.
        * The "in" operator is implemented once and works for each subclass
    """
    def __init__(
            self, 
            center: np.ndarray,                     # center of the circle
            radius: float,                          # radius of the circle
            norm:   Callable[[np.ndarray], float]   # a function IR^d x IR^d --> IR_{>= 0}
        ) -> None:
        assert radius >= 0

        ## Dimensions
        self.row_dim    = center.shape[0]
        self.column_dim = center.shape[1]

        ## Radius
        self.radius = radius

        ## Center
        self.center = center

        ## Norm
        self.norm = norm


    ## Accessors
    def get_radius(self) -> float:
        """
            Return the radius
        """
        return self.radius
    
    def get_center(self) -> np.ndarray:
        """
            Return the radius
        """
        return self.center
    
    ## Mutators
    def set_radius(self, new_radius:float):
        """
            Set radius
        """
        assert new_radius >= 0

        self.radius = new_radius
    
    def set_center(self, new_center:np.ndarray):
        """
            Set the center
        """
        assert new_center.shape[0] == self.row_dim
        assert new_center.shape[1] == self.column_dim

        self.center = new_center
    
    ## Predicates
    def __contains__(self, x:np.ndarray):
        """
            Check if x \in C, i.e.,
            ``||x - c ||_+oo <= r``.
        """
        assert x.shape[0] == self.row_dim
        assert x.shape[1] == self.column_dim

        #######################################################################
        # Why epsilon?
        # --------------------------------------------------------------------
        # * We use the epsilon constant defined in geometry.constants
        # * We use this percision constant in order to avoid paradoxes
        # of Marabou computing counterexamples, *not* belonging to
        # the explanation.
        # * Marabou counterexamples become misclassified by the explanation
        # due to differences in the ~16th decimal point.
        # * Using epsilon we only consider differences up to the 8th decimal
        # point.
        #######################################################################
        return self.norm(x - self.center) <= self.radius + epsilon




###########################################################
# Class: InfCircle
# --------------------------------------------------------
# * The infinity-norm circle.
# * Since inf-circle is a box in IR^d, we implement a
# conversion method
###########################################################
class InfCircle(Circle):
    """
        * The infinity-norm circle.
        * Since inf-circle is a box in IR^d, we implement a conversion method
    """
    ## Initialization
    def __init__(self, center: np.ndarray, radius: float) -> None:

        ## Initialize super class
        super().__init__(center, radius, norms.inf_norm)

    ## Conversions
    def get_interval(self) -> interval.Interval:
        """
            Convert the l_+oo circle to interval. Note that
            l_+oo circles correspond to hyper-cubes.
        """
        return interval.Interval(
            self.center - self.radius * np.ones((self.row_dim, self.column_dim)),
            self.center + self.radius * np.ones((self.row_dim, self.column_dim))
        )



