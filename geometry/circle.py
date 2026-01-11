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
    def __init__(
            self, 
            center,     # center of the circle
            radius,     # radius of the circle
            norm        # a function IR^d x IR^d --> IR_{>= 0}
        ):
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
    def get_radius(self):
        return self.radius
    
    def get_center(self):
        return self.radius
    
    ## Mutators
    def set_radius(self, new_radius):
        assert new_radius >= 0

        self.radius = new_radius
    
    def set_center(self, new_center):
        assert new_center.shape[0] == self.row_dim
        assert new_center.shape[1] == self.column_dim

        self.center = new_center
    
    ## Predicates
    def __contains__(self, x):
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
    ## Initialization
    def __init__(self, center, radius):

        ## Initialize super class
        super().__init__(center, radius, norms.inf_norm)

    ## Conversions
    def get_interval(self):
        return interval.Interval(
            self.center - self.radius * np.ones((self.row_dim, self.column_dim)),
            self.center + self.radius * np.ones((self.row_dim, self.column_dim))
        )



