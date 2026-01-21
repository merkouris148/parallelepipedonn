## Python libraries
# Importing parent directory class
# see: https://www.geeksforgeeks.org/python-import-from-parent-directory/
import sys
sys.path.append('..')
import geometry.interval as interval
import geometry.circle as circle
from geometry.norms import inf_norm

## 3rd party libraries
import numpy as np


###########################################################
# class: CyclicGuarantee
# --------------------------------------------------------
# A class implementing cyclic guarantees, w.r.t. the
# \ell_\infty circle.
# 
# These guarantees take the form, [x* - r1, x* + r1], and
# are equivalent to adversarial robustness notions, of
# existing work.
#
# Consider the domain IF. If [x* - r1, x* + r1] \ IF !=
# \emptyset, then we take the intersection of the guarantee
# with the domain IF.
###########################################################
class CyclicGuarantee(circle.InfCircle):
    ## Initialization
    # Constructor
    def __init__(
                    self,
                    x_star,                 # the input point
                    c_star,                 # the predicted class of x_star
                    distance_restriction,   # the maximum distance of the guarantee's
                                            # radius
                    delta,                  # the step of the expand/constain 
                                            # operations
                    current_radius,         # the initial radius of the guarantee,
                                            #   * cr = dr, for the top guarantee,
                                            #   * cr = 0, for the bottom guarantee
                    domain                  # the input domain IF, is given as
                                            # a geometry.Interval instance
                ):
        
        ## Initialize super-class
        super().__init__(x_star, current_radius)
        
        assert delta > 0
        assert distance_restriction >= delta

        ## Initialize
        self.c_star                 = c_star
        self.delta                  = delta
        self.distance_restriction   = distance_restriction
        self.domain                 = domain

        self.pivot                  = interval.Interval(
                                        np.array([[0]]),
                                        np.array([[self.distance_restriction]])
                                    )
    

    # Copy constructor
    def __copy__(self):
        return CyclicGuarantee(
            self.center,
            self.c_star,
            self.distance_restriction,
            self.delta,
            self.radius,
            self.domain
        )

    ## Operations
    
    ###########################################################
    # CyclicGuarantee.constain()
    # --------------------------------------------------------
    # Let [x* - r1, x* + r1] be the current guarantee and
    # x^c \in [x* - r1, x* + r1] a counter example.
    # Let l = \ell_\infty(x* - x^c) then r' = l - \delta is
    # the radius of the new guarantee.
    ###########################################################
    def constrain(self, counterexample):
        ## New radius
        val = inf_norm(counterexample - self.center)
        if val - self.delta > 0:
            self.set_radius(val - self.delta)
            return True
        else:
            return False
    

    ###########################################################
    # CyclicGuarantee.expand()
    # --------------------------------------------------------
    # Let [x* - r1, x* + r1] be the current guarantee, then
    # r' = r + \delta is the radius of the new guarantee.
    # 
    # Note that this operation may lead to an *unsound*
    # guarantee, i.e. by expandin the guarantee may include a
    # counter example.
    ###########################################################
    def expand(self):
        if self.get_radius() + self.delta < self.distance_restriction:
            self.set_radius(self.get_radius() + self.delta)
            return True
        else:
            return False


    ###########################################################
    # CyclicGuarantee.expand_dichotomic()
    # --------------------------------------------------------
    # Precondition:
    #   * r \in [pivot.lb, pivot.ub],
    #   with either
    #       r = pivot.lb
    #   or
    #       r = pivot.ub
    #
    # Postcondition:
    #   * r' = pivot.lb + (pivot.ub - pivot.lb)/2
    ###########################################################
    def expand_dichotomic(self):
        new_radius = self.pivot.lb[0][0] + (self.pivot.ub[0][0] - self.pivot.lb[0][0])/2

        if self.get_radius() + self.delta < self.distance_restriction:
            self.set_radius(new_radius)
            return True
        else:
            return False
    


    ###########################################################
    # CyclicGuarantee.up_pivot()
    # --------------------------------------------------------
    # Precondition:
    #   * r \in [pivot.lb, pivot.ub],
    #   with
    #       r = pivot.lb + (pivot.ub - pivot.lb)
    #
    # Postcondition:
    #   * pivot.lb = r
    #
    # We use this operation when the current guarantee IS
    # sound. Since the guarantee is sound, then there is NO
    # counterexample in [x* - r1, x* + r1]. Thus, there is NO
    # counterexample in [x* - (pivot.lb)1, x* + (pivot.lb)1].
    # Hence, the values in [pivot.lb, r] are unecessary
    # restrictive. Therefore we raise the pivot.lb to r.
    ###########################################################
    def up_pivot(self):
        tmp = self.pivot.lb
        self.pivot.lb = np.array([[self.radius]])

        if self.pivot.inequalities_consistency():
            return True
        else:
            self.pivot.lb = tmp
            return False



    ###########################################################
    # CyclicGuarantee.down_pivot()
    # --------------------------------------------------------
    # Precondition:
    #   * r \in [pivot.lb, pivot.ub],
    #   with
    #       r = pivot.lb + (pivot.ub - pivot.lb)
    #
    # Postcondition:
    #   * pivot.ub = r
    #
    # We use this operation when the current guarantee is NOT
    # sound. If [x* - r1, x* + r1] the current guarantee and
    # there is a counterexample x^c \in [x* - r1, x* + r1],
    # then there will also be a counterexample
    # x^c \in [x* - (pivot.ub)1, x* + (pivot.ub)1]. Thus, we
    # need to lower the pivot.ub.
    ###########################################################
    def down_pivot(self):
        tmp = self.pivot.ub
        self.pivot.ub = np.array([[self.radius]])

        if self.pivot.inequalities_consistency():
            return True
        else:
            self.pivot.ub = tmp
            return False
    

    ###########################################################
    # CyclicGuarantee.dichotomic_invariant()
    # --------------------------------------------------------
    # pivot.ub - pivot.lb >= \delta
    ###########################################################
    def dichotomic_invariant(self):
        return (self.pivot.ub[0][0] - self.pivot.lb[0][0]) >= self.delta


    ###########################################################
    # CyclicGuarantee.make_sound()
    # --------------------------------------------------------
    # It always hold that [x* - (pivot.lb)1, x* + (pivot.lb)1]
    # is ALWAYS sound. Thus, we assign,
    #   r = pivot.lb
    ###########################################################
    def make_sound(self):
        self.radius = self.pivot.lb.copy()
    
    ###########################################################
    # CyclicGuarantee.make_sound()
    # --------------------------------------------------------
    # It always hold that [x* - (pivot.lb)1, x* + (pivot.lb)1]
    # is ALWAYS sound. Thus, we assign,
    #   r = pivot.lb
    ###########################################################
    def make_complete(self):
        self.radius = self.pivot.ub.copy()

    ###########################################################
    # CyclicGuarantee.get_interval()
    # --------------------------------------------------------
    # We overload the super-class' operation. Let IF be the
    # input domain of the neural network. We return the
    # intersection, [x* - r1, x* + r1] \cap IF.
    ###########################################################
    def get_interval(self):
        inf_circle = super().get_interval()
        inf_circle.intersect(self.domain)

        return inf_circle
    


    ## Metrics

    def calc_complexity(self):
        inf_circle = super().get_interval()
        inf_circle.intersect(self.domain)

        return np.sum(inf_circle.lb < self.center) + np.sum(inf_circle.ub > self.center)


    def calc_max_inf_radius(self):
        inf_circle = super().get_interval()
        inf_circle.intersect(self.domain)

        return np.min(inf_circle.ub - inf_circle.lb)


###########################################################
# Class: TopCyclicGuarantee
# --------------------------------------------------------
# The top (most general) cyclic guarantee, with
#
#   current_radius = distance_restriction
###########################################################
class TopCyclicGuarantee(CyclicGuarantee):
    def __init__(self, x_star, c_star, distance_restriction, delta, domain):
        super().__init__(
                            x_star,
                            c_star,
                            distance_restriction,
                            delta,
                            distance_restriction,
                            domain
                        )




###########################################################
# Class: BottomCyclicGuarantee
# --------------------------------------------------------
# The bottom (most specific) cyclic guarantee, with
#
#   current_radius = 0
###########################################################
class BottomCyclicGuarantee(CyclicGuarantee):
    def __init__(self, x_star, c_star, distance_restriction, delta, domain):
        super().__init__(
                            x_star,
                            c_star,
                            distance_restriction,
                            delta,
                            0,
                            domain
                        )