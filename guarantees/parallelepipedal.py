## Python libraries
from copy import copy
import typing

# Importing parent directory class
# see: https://www.geeksforgeeks.org/python-import-from-parent-directory/
import sys
#sys.path.append('..')
import geometry.interval as interval
import geometry.circle as circle
#from geometry.constants import epsilon

## 3rd party libraries
import numpy as np

class ParallelepipedalGuarantee(interval.Interval):
    """
        #### Description:
        A class encoding a stability guarantee. Essentially
        a stability guarantee is an *interval* of the form
        `[lb, ub]`, where `lb, ub` are *vectors* of the
        `domain` input space.

        #### Guarantee Refinement Operations:
        * `constrain(x^c)`: Excludes the counterexample x^c from the guarantee.
        * `expand()`: Expands the guarantee (every coordinate of lb, ub) by delta.
        * `expand_ub(i, j)`: Expand by delta only the (i,j)-th coordinate of ub.
        * `expand_lb(i, j)`: Expand by delta only the (i,j)-th coordinate of lb.
        * `revert_expand_ub(i, j)`: Reduces by delta the (i, j)-th coordinate of ub.
        * `revert_expand_lb(i, j)`: Reduces by delta the (i, j)-th coordinate of lb.
    """

    ###############
    # Constructor #
    ###############

    def __init__(
            self,
            x_star: np.ndarray,
            c_star: int,
            delta:  float,
            domain: interval.Interval # an interval
        ) -> None:
        ## Initializing super-class with the whole IR^d
        interval.Interval.__init__(self, -np.inf * np.ones(x_star.shape), np.inf * np.ones(x_star.shape))
        assert x_star in domain

        ## Initialization
        self.domain     = domain
        self.x_star     = x_star
        self.c_star     = c_star
        self.delta      = delta
        self.row_dim    = x_star.shape[0]
        self.column_dim = x_star.shape[1]
        self.dim        = self.row_dim * self.column_dim


        ## ONLY for dichotomic search
        self.high_pivot = interval.Interval(self.x_star.copy(), self.domain.ub.copy())
        self.low_pivot  = interval.Interval(self.domain.lb.copy(), self.x_star.copy())
    

    # Copy constructor
    def __copy__(self):
        guarantee = ParallelepipedalGuarantee(
            self.x_star,
            self.c_star,
            self.delta,
            self.domain
        )

        guarantee.ub = self.ub.copy()
        guarantee.lb = self.lb.copy()

        ## Dichotomic search
        guarantee.high_pivot = copy(self.high_pivot)
        guarantee.low_pivot = copy(self.low_pivot)

        return guarantee

    
    ############
    # Mutators #
    ############

    def set_bounds(
            self,
            new_lb: typing.Union[np.ndarray, None],
            new_ub: typing.Union[np.ndarray, None]
        ) -> typing.Tuple[bool, bool]:
        """
            #### Description:
            Setting the bounds of a *parallelepipedal* guarantee.
        """

        lb_set = False
        if new_lb is not None:
            assert new_lb.shape == (self.row_dim, self.column_dim)
            assert (new_lb <= self.x_star).all()
            assert (self.domain.lb <= new_lb).all()

            self.lb = new_lb.copy()
            lb_set  = True
        

        ub_set = False
        if new_ub is not None:
            assert new_ub.shape == (self.row_dim, self.column_dim)
            assert (self.x_star <= new_ub).all()
            assert (new_ub <= self.domain.ub).all()

            self.ub = new_ub.copy()
            ub_set  = True
        
        ## When changing bounds we NEED to update the pivots!
        self.update_pivots()

        return lb_set, ub_set


    ##################################
    # Operations for Top Down Search #
    ##################################
    
    # Select inequality to update
    def select_inequality(self, counter_example: np.ndarray) -> typing.Tuple[int, bool]:
        ## upper difference
        diff        = counter_example - self.x_star
        abs_diff    = np.abs(diff)
        ind         = np.unravel_index(abs_diff.argmax(), abs_diff.shape)

        ## Choose to update lower or upper bounds
        update_ub = diff[ind] > 0

        ## return values
        return ind, update_ub


    # Refine explanation, given a counter example
    def constrain(self, witness: np.ndarray) -> bool:
        assert witness in self
        ## Sanity check
        old_potential = self.calc_potential()

        ind, update_ub = self.select_inequality(witness)

        ## Updating inequalities
        if update_ub:
            if witness[ind] - self.delta < self.x_star[ind]: return False
            self.update_ub(
                ind,
                witness[ind] - self.delta
            )
        else:
            if witness[ind] + self.delta > self.x_star[ind]: return False
            self.update_lb(
                ind,
                witness[ind] + self.delta
            )
        
        ## Sanity check
        new_potential = self.calc_potential()
        assert old_potential - new_potential > 0
        
        
        ## Constrain successful
        return True

    ##################################
    # Sequential Generalization Algo #
    ##################################
    def generalize(self, witness: np.ndarray) -> bool:
        assert not (witness in self)
        ## Sanity check
        old_potential = self.calc_potential()


        witness_interval = interval.Interval(
            witness - self.delta * np.ones((self.row_dim, self.column_dim)),
            witness + self.delta * np.ones((self.row_dim, self.column_dim))
        )
        # we need to intersect with the domain,
        # otherwise we'll go beyonds its bounds,
        # and the potential won't make sense, i.e.
        # we'll have potential > 1.
        witness_interval.intersect(self.domain)
        self.concatenate(witness_interval)
        
        ## Sanity check
        new_potential = self.calc_potential()
        assert new_potential - old_potential > 0
        
        ## Constrain successful
        return not self.includes(self.domain)
    
    ###################################
    # Operations for Bottom Up Search #
    ###################################
    
    def expand(self) -> bool:
        ## Check that the explanation does not exceed the domain
        if (self.ub + self.delta * np.ones([self.row_dim, self.column_dim]) > self.domain.ub).any() or\
            (self.lb - self.delta * np.ones([self.row_dim, self.column_dim]) < self.domain.lb).any():
            
            return False

        ## compute expantion
        self.ub += self.delta * np.ones([self.row_dim, self.column_dim])
        self.lb -= self.delta * np.ones([self.row_dim, self.column_dim])

        return self.inequalities_consistency()

    
    #######################################
    # Operations for Bottom Up Linear DFS #
    #######################################
    
    # expand a feature by delta
    def expand_ub(self, i, j):
        ## Check that the explanasion does not exceed the domain
        if self.ub[i][j] + self.delta > self.domain.ub[i][j]: return False

        ## compute expansion
        self.update_ub((i, j), self.ub[i][j] +  self.delta)
        return True

    def expand_lb(self, i, j):
        ## Check that the explanasion does not exceed the domain
        if self.lb[i][j] -  self.delta < self.domain.lb[i][j]: return False

        ## compute expansion
        self.update_lb((i, j), self.lb[i][j] -  self.delta)
        return True
    
    # revert a previous expansion
    def revert_expand_ub(self, i, j):
        self.ub[i][j] -= self.delta

    def revert_expand_lb(self, i, j):
        self.lb[i][j] += self.delta



    ###########################################
    # Operations for Bottom Up Dichotomic DFS #
    ###########################################

    def expand_dichotomic_ub(self, i: int, j: int) -> bool:
        """ 
            #### Description:
            The expand `ub` operation used in Bottom-Up Dichotomic DFS.

            #### Percondition:
            `ub` in `[high_pivot.lb, high_pivot.ub]`, with either
            * `ub = high_pivot.lb`, or
            * `ub = high_pivot.ub`
    
            #### Postcondition:
            * `ub = high_pivot.lb + (high_pivot.ub - high_pivot.lb)/2`
            * the precondition still holds.

            #### Output:
            * `False` is returned iff the expnasion does not exceed the domain.
            Namely, iff `new_ub_ij > dom_ub_ij`
        """

        new_ub_ij = self.high_pivot.lb[i][j] + (self.high_pivot.ub[i][j] - self.high_pivot.lb[i][j]) / 2
        if new_ub_ij > self.domain.ub[i][j]: return False
        
        self.update_ub((i, j), new_ub_ij)
        return True


    def expand_dichotomic_lb(self, i: int, j: int) -> bool:
        """
            #### Description:
            The expand `lb` operation used in Bottom-Up Dichotomic DFS.

            #### Percondition:
            `lb` in `[low_pivot.lb, low_pivot.ub]`, with either:
            * `lb = low_pivot.lb`, or
            * `lb = low_pivot.ub`
    
            #### Postcondition:
            * `lb = low_pivot.lb + (low_pivot.ub - low_pivot.lb)/2`
            * the precondition still holds.

            #### Output:
            * `False` is returned iff the expnasion does not exceed the domain.
            Namely, iff `new_lb_ij < dom_lb_ij`
        """

        new_lb_ij = self.low_pivot.lb[i][j] + (self.low_pivot.ub[i][j] - self.low_pivot.lb[i][j]) / 2
        if new_lb_ij < self.domain.lb[i][j]: return False
        
        self.update_lb((i, j), new_lb_ij)
        return True


    ## Refine high_pivot

    def down_high_pivot(self, i: int, j: int) -> bool:
        """
            #### Description:
            Refinement of the `high_pivot`, used in Bottom-Up Dichtomic DFS.

            #### Percondition:
            `ub` in `[high_pivot.lb, high_pivot.ub]`, with
            * `ub = high_pivot.lb + (high_pivot.ub - high_pivot.lb)/2`
    
            #### Postcondition:
            * `high_pivot.ub = ub`
    
            #### Notes:
            We use this operation when the current guarantee is NOT
            sound. If `[lb, ub]` the current guarantee and there is a
            counterexample `x^c` in `[lb, ub]`, then there will also
            be a counterexample `x^c` in `[lb, high_pivot.ub]`. Thus, we
            need to lower the `high_pivot.ub`.
        """

        tmp = self.high_pivot.ub[i][j]
        self.high_pivot.ub[i][j] = self.ub[i][j]

        if self.high_pivot.inequality_consistency((i, j)):
            return True

        else:
            self.high_pivot.ub[i][j] = tmp
            return False


    ###########################################################
    # ParallelepipedalGuarantee.up_high_pivot()
    # --------------------------------------------------------
    # Percondition:
    #   * ub \in [high_pivot.lb, high_pivot.ub],
    #   with
    #       ub = high_pivot.lb + (high_pivot.ub - high_pivot.lb)/2
    #
    # Postcondition:
    #   * high_pivot.lb = ub
    #
    # We use this operation when the current guarantee IS
    # sound. Since the guarantee is sound, then there is NO
    # counterexample in [lb, ub]. Thus, there is NO
    # counterexample in [high_pivot.lb, ub]. Hence, the values
    # in [high_pivot.lb, ub] are unecessary restrictive.
    # Therefore we raise the high_pivot.lb to ub.
    ###########################################################
    def up_high_pivot(self, i, j):
        #self.high_pivot.update_lb((i,j), self.ub[i][j])
        tmp = self.high_pivot.lb[i][j]
        self.high_pivot.lb[i][j] = self.ub[i][j]

        if self.high_pivot.inequality_consistency((i, j)):
            return True

        else:
            self.high_pivot.lb[i][j] = tmp
            return False


    ## Refine low_pivot:

    ###########################################################
    # ParallelepipedalGuarantee.down_low_pivot()
    # --------------------------------------------------------
    # Percondition:
    #   * lb \in [low_pivot.lb, low_pivot.ub],
    #   with
    #       lb = low_pivot.lb + (low_pivot.ub - low_pivot.lb)/2
    #
    # Postcondition:
    #   * low_pivot.lb = lb
    #
    # We use this operation when the current guarantee IS
    # sound. Since the guarantee is sound, then there is NO
    # counterexample in [lb, ub]. Thus, there is NO
    # counterexample in [lb, low_pivot.ub]. Hence, the values
    # in [lb, low_pivot.ub] are unecessary restrictive.
    # Therefore we lower the low_pivot.ub to lb.
    ###########################################################
    def down_low_pivot(self, i, j):
        #self.low_pivot.update_ub((i, j), self.lb[i][j])
        tmp = self.low_pivot.ub[i][j]
        self.low_pivot.ub[i][j] = self.lb[i][j]

        if self.low_pivot.inequality_consistency((i, j)):
            return True

        else:
            self.low_pivot.ub[i][j] = tmp
            return False


    ###########################################################
    # ParallelepipedalGuarantee.up_low_pivot()
    # --------------------------------------------------------
    # Percondition:
    #   * lb \in [low_pivot.lb, low_pivot.ub],
    #   with
    #       lb = low_pivot.lb + (low_pivot.ub - low_pivot.lb)/2
    #
    # Postcondition:
    #   * high_pivot.lb = ub
    #
    # We use this operation when the current guarantee is
    # NOT sound. Thus, there is a counterexample
    # x^c in [lb, ub]. Hence there is a counterexample in
    # [low_pivot.lb, ub]. Therefore, we raise low_pivot.lb to lb.
    ###########################################################
    def up_low_pivot(self, i, j):
        #self.low_pivot.update_lb((i, j), self.lb[i][j])
        tmp = self.low_pivot.lb[i][j]
        self.low_pivot.lb[i][j] = self.lb[i][j]

        if self.low_pivot.inequality_consistency((i, j)):
            return True

        else:
            self.low_pivot.lb[i][j] = tmp
            return False
    

    ## Invariants for continuing dichotomic search

    ###########################################################
    # ParallelepipedalGuarantee.high_dichotomic_invariant()
    # --------------------------------------------------------
    # high_pivot.ub - high_pivot.lb >= \delta
    ###########################################################
    def high_dichotomic_invariant(self, i, j):
        return (self.high_pivot.ub[i][j] - self.high_pivot.lb[i][j]) >= self.delta

    ###########################################################
    # ParallelepipedalGuarantee.low_dichotomic_invariant()
    # --------------------------------------------------------
    # low_pivot.ub - low_pivot.lb >= \delta
    ###########################################################
    def low_dichotomic_invariant(self, i, j):
        return (self.low_pivot.ub[i][j] - self.low_pivot.lb[i][j]) >= self.delta

    ## Revert [lb, ub] to the last "safe" position
    # only to be used with dichotomic search

    ###########################################################
    # ParallelepipedalGuarantee.make_sound()
    # --------------------------------------------------------
    # It always hold that [low_pivot.ub, high_pivot.lb] is
    # ALWAYS sound. Thus, we assign,
    #   [lb, ub] = [low_pivot.ub, high_pivot.lb]
    ###########################################################
    def make_sound(self):
        self.ub = self.high_pivot.lb.copy()
        self.lb = self.low_pivot.ub.copy()
    
    ## For algorithm composition
    # before passing the explanation from the top-down
    # algorithm to the bottom-up, we need to update the
    # pivots
    def update_pivots(self):
        self.high_pivot = interval.Interval(self.ub.copy(), self.domain.ub.copy())
        self.low_pivot  = interval.Interval(self.domain.lb.copy(), self.lb.copy())

    
    ###########
    # Metrics #
    ###########
    
    def calc_complexity(self):
        return np.sum(self.lb < self.x_star) + np.sum(self.ub > self.x_star)




class TopParallelGuarantee(ParallelepipedalGuarantee):
    ## Constructor
    def __init__(
            self,
            x_star,
            c_star,
            delta,
            domain
        ):
        ## Initializing super-class with the whole IR^d
        super().__init__(x_star, c_star, delta, domain)

        ## Setting the internal interval equal to the domain
        self.intersect(self.domain)




class BottomParallelGurantee(ParallelepipedalGuarantee):
    ## Constructor
    def __init__(
            self,
            x_star,
            c_star,
            delta,
            domain
        ):
        ## Initializing super-class with the whole IR^d
        super().__init__(x_star, c_star, delta, domain)

        ## Setting the internal interval equal to x_star
        self.intersect(interval.Interval(self.x_star, self.x_star))




#####################
# Distance Inflated #
#####################
class DistParallelGurantee(ParallelepipedalGuarantee):
    ## Constructor
    def __init__(
            self,
            x_star,
            c_star,
            radius,
            delta,
            domain,
        ):
        ## Initializing super-class with the whole IR^d
        super().__init__(
            x_star,
            c_star,
            delta,
            domain
        )

        ## Distance restriction
        self.radius = radius
        self.domain.intersect(circle.InfCircle(self.x_star, self.radius).get_interval())



class TopDistParallelGurantee(DistParallelGurantee):
    ## Constructor
    def __init__(
            self,
            x_star,
            c_star,
            radius,
            delta,
            domain
        ):
        ## Initializing super-class with the whole IR^d
        super().__init__(x_star, c_star, radius, delta, domain)

        ## Setting the internal interval equal to the domain
        self.intersect(self.domain)



class BottomDistParallelGurantee(DistParallelGurantee):
    ## Constructor
    def __init__(
            self,
            x_star,
            c_star,
            radius,
            delta,
            domain
        ):
        ## Initializing super-class with the whole IR^d
        super().__init__(x_star, c_star, radius, delta, domain)

        ## Setting the internal interval equal to x_star
        self.intersect(interval.Interval(self.x_star, self.x_star))
