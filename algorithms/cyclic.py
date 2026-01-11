#############
# Libraries #
#############

# libraries for typing
import typing
from verification.nn_verification import NNVerification

# custom libraries
from algorithms.algorithms import SearchAlgorithm
import guarantees.cyclic as cyclic

# python libraries
import time
from copy import copy



class CyclicSearch(SearchAlgorithm):
    """
        #### Description:
        The top class of search algorithms for cyclic guarantees.
        Each of the inherited classes needs to define a search method.
        This search method should take as input a cyclic guarantee.
    """
    def __init__(
            self,
            isSAT:      NNVerification,
            max_it:     int = 1000,
            verbose:    bool = False
        ) -> None:
        
        super().__init__(isSAT, max_it, verbose)

    def search(
            self,
            guarantee: cyclic.CyclicGuarantee
        ) -> cyclic.CyclicGuarantee:

        raise NotImplementedError



###################
# Top-Down Search #
###################

# This method is *essentially* copy-paste from
# algos.parallelepipedal.TopDownSearch.
# Naturally, this is not a good programming practice,
# I copy-pasted this algo here only to change the
# *type*.
class TopDownSearch(CyclicSearch):
    """
        #### Description:
        * A search algorithm implementing a top down search in
        the guarantee space.
        * The guarantee passed to the search() method needs to
        have defined a constrain() method that takes as input a
        counterexample and returns a refined guarantee that
        excludes the given counterexample.
    """

    def __init__(
                    self,
                    isSAT,
                    max_it = 1000,   # max number of iterations
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Cyc. Top-Down " + self.msg_prefix
    

    def search(
            self,
            guarantee: cyclic.TopCyclicGuarantee
        ) -> cyclic.CyclicGuarantee:
        # time
        tic = time.time()

        # main loop
        # num_it counts the number of oracle calls
        for self.num_it in range(self.max_it):
            ## Check convergance
            self.soundness, counterexample = self.isSAT(guarantee.get_interval())
            if self.soundness: break

            ## Refine the explanation
            self.refinement_success = guarantee.constrain(counterexample)
            if not self.refinement_success: break

            ## Reporting
            self.progress_message()

        # time
        toc = time.time()
        self.total_time = toc - tic

        ## Warning
        self.end_report()

        # return value
        return guarantee




#####################
# Bottom-Up Methods #
#####################


class BottomUpLinearSearch(CyclicSearch):
    """
        #### Description:
        * A search algorithm implementing a bottom up search in
        the space of guarantees.
        * The guarantee passed to the search() method needs to
        have defined an `expand()` method.
        * Primarly used for cyclic guarantees
    """

    def __init__(
                    self,
                    isSAT,
                    max_it = 1000,   # max number of iterations
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Cyc. Bottom-Up " + self.msg_prefix
    

    def search(
            self,
            guarantee: cyclic.BottomCyclicGuarantee
        ) -> cyclic.CyclicGuarantee:
        # time
        tic = time.time()

        # main loop
        # num_it *does not* counts the number of
        # isSAT oracle calls here
        for self.num_it in range(self.max_it):
            ## Keep the old guarantee, in case expansion does not work
            old_guarantee = copy(guarantee)

            ## Refine the guarantee
            self.refinement_success = guarantee.expand()
            if not self.refinement_success: break

            ## Check convergance
            self.soundness, _ = self.isSAT(guarantee.get_interval())
            if not self.soundness:
                # Since we start with the trivial guarantee and expand
                # the guarantee will always be sound, until a counter
                # example is provided. If that hapens, we revert to the
                # previous sound guarantee.
                self.soundness = True
                guarantee = old_guarantee
                break

            ## Reporting
            self.progress_message()

        # time
        toc = time.time()
        self.total_time = toc - tic

        ## Warning
        self.end_report()

        # return value
        return guarantee



class BottomUpDichotomicSearch(CyclicSearch):
    """
        #### Desrciption:
        * A search algorithm implementing a bottom up dichotomic
        search in the space of guarantees.
        * The guarantee passed to the search() method needs to
        have defined an `expand_dichotomic()` method.
        * Primarly used for cyclic guarantees
    """
    def __init__(
                    self,
                    isSAT,
                    max_it = 1000,   # max number of iterations
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Cyc. Bottom-Up Dich. " + self.msg_prefix
    

    def search(
            self,
            guarantee: cyclic.BottomCyclicGuarantee
        ) -> cyclic.CyclicGuarantee:
        # time
        tic = time.time()

        # main loop
        # expand *upper bound* with linear search
        for it in range(self.max_it):
            ## if dichotomic search converged, break
            if not guarantee.dichotomic_invariant(): break


            ## Refine the explanation
            self.refinement_success = guarantee.expand_dichotomic()
            if not self.refinement_success: break

            ## Reporting
            self.num_it += 1
            self.progress_message()

            ## Check convergance
            self.soundness, _ = self.isSAT(guarantee.get_interval())
            if self.soundness:
                succ_pivot_refinement = guarantee.up_pivot()
                if not succ_pivot_refinement: break
            
            else:
                succ_pivot_refinement = guarantee.down_pivot()
                if not succ_pivot_refinement: break


        ## peculiarity of dichotomic search
        if not self.soundness:
            guarantee.make_sound()
            self.soundness = True


        # time
        toc = time.time()
        self.total_time = toc - tic

        ## Warning
        self.end_report()

        # return value
        return guarantee