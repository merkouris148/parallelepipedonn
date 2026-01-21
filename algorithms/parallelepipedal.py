#############
# Libraries #
#############
# custom libraries
from algorithms.algorithms import SearchAlgorithm

# libraries for typing
import typing
from verification.nn_verification import NNVerification
import guarantees.parallelepipedal as parallel

# python libraries
import time
from copy import copy


class ParallelepipedalSearch(SearchAlgorithm):
    """
        #### Description:
        The top class of search algorithms for parallelepipedal guarantees.
        Each of the inherited classes needs to define a search method.
        This search method should take as input a parallelepipedal guarantee.
    """

    def __init__(
            self,
            isSAT:      NNVerification,
            max_it:     int = 1000,
            timeout:    int = 60,
            verbose:    bool = False
        ):
        super().__init__(isSAT, max_it, timeout, verbose)
    
    def search(
            self,
            guarantee:  parallel.ParallelepipedalGuarantee
        ) -> parallel.ParallelepipedalGuarantee:
        
        raise NotImplementedError


####################
# Top-Down Methods #
####################

class TopDownSearch(ParallelepipedalSearch):
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
                    max_it  = 1000,   # max number of iterations
                    timeout = 60,
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, timeout, verbose)

        ## Reporting
        self.msg_prefix = "Top-Down " + self.msg_prefix
        self.prop_name  = "Soundness"
    

    def search(
            self,
            guarantee: typing.Union[
                parallel.TopParallelGuarantee,
                parallel.TopDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        self.timer_start()

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

            ## Check Timeout
            if self.check_timeout(): break


        # time
        self.timer_stop()

        ## Warning
        self.end_report()

        # return value
        return guarantee


#####################
# Comleteness Algos #
#####################

class CompleteBottomUpSearch(ParallelepipedalSearch):
    """
        #### Description:
        * A search algorithm implementing a bottom up search in
        the guarantee space.
        * computing a complete approximation
        * The guarantee passed to the search() method needs to
        have defined a concatenate() method that takes as input a
        counterexample and returns a refined guarantee that
        excludes the given counterexample.
    """

    def __init__(
                    self,
                    isSAT,
                    max_it = 1000,   # max number of iterations
                    timeout = 60,
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, timeout, verbose)

        ## Reporting
        self.msg_prefix = "Complete Bottom-Up " + self.msg_prefix
        self.prop_name  = "Completeness"


    def search(
            self,
            guarantee: typing.Union[
                parallel.BottomParallelGurantee,
                parallel.BottomDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        self.timer_start()

        # main loop
        # num_it counts the number of oracle calls
        for self.num_it in range(self.max_it):
            ## Check convergance
            self.completeness, witness = self.isSAT(guarantee.get_interval())
            if self.completeness: break

            ## Refine the explanation
            self.refinement_success = guarantee.generalize(witness)
            if not self.refinement_success: break

            ## Reporting
            self.progress_message()

            ## Check Timeout
            if self.check_timeout(): break


        # time
        self.timer_stop()

        ## Warning
        self.end_report()

        # return value
        return guarantee


#####################
# Bottom-Up Methods #
#####################


class BottomUpLinearDFS(ParallelepipedalSearch):
    """
        #### Description:
        * A search algorithm implementing a bottom up linear search in
        the space of guarantees.
        * Expanding one feature at the time.
        * The guarantee passed to the search() method needs to
        have defined the following methods:
            * `expand_ub(i, j)`
            * `expand_lb(i, j)`
            * `revert_expand_ub(i, j)`
            * `revert_expand_lb(i, j)`
        
        #### See also:
        Used in the paper: "Delivering Inflated Explanations" - Y. Izza et al. (2024)
    """

    def __init__(
                    self,
                    isSAT,
                    max_it = 100,   # max number of iterations
                    timeout = 60,
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, timeout, verbose)

        ## Reporting
        self.msg_prefix = "Bottom-Up Lin. DFS"
        self.prop_name  = "Soundness"
    

    def search(
            self,
            guarantee: typing.Union[
                parallel.BottomParallelGurantee,
                parallel.BottomDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        self.timer_start()

        # main loop
        # expand *upper bound* with linear search
        for i in range(guarantee.row_dim):
            for j in range(guarantee.column_dim):
                for it in range(self.max_it):
                    ## Keep the old explanation, in case expansion does not work
                    #old_explanation = copy(explanation)

                    ## Refine the explanation
                    self.refinement_success = guarantee.expand_ub(i, j)
                    if not self.refinement_success: break

                    ## Reporting
                    self.num_it += 1
                    self.progress_message()

                    ## Check convergance
                    self.soundness, _ = self.isSAT(guarantee.get_interval())
                    if not self.soundness:
                        # Since we start with the trivial explanation and expand
                        # the explanation will always be sound, until a counter
                        # example is provided. If that hapens, we revert to the
                        # previous sound explanation.
                        self.soundness = True
                        guarantee.revert_expand_ub(i, j)
                        break
                    
                    # time
                    if self.check_timeout(): break
                if self.is_timeout: break
            if self.is_timeout: break


        # expand *lower bound* with linear search
        if not self.is_timeout:
            for i in range(guarantee.row_dim):
                for j in range(guarantee.column_dim):
                    for it in range(self.max_it):
                        ## Keep the old explanation, in case expansion does not work
                        #old_explanation = copy(explanation)

                        ## Refine the explanation
                        self.refinement_success = guarantee.expand_lb(i, j)
                        if not self.refinement_success: break

                        ## Reporting
                        self.num_it += 1
                        self.progress_message()

                        ## Check convergance
                        self.soundness, _ = self.isSAT(guarantee.get_interval())
                        if not self.soundness:
                            # Since we start with the trivial explanation and expand
                            # the explanation will always be sound, until a counter
                            # example is provided. If that hapens, we revert to the
                            # previous sound explanation.
                            self.soundness = True
                            guarantee.revert_expand_lb(i, j)
                            #explanation = old_explanation
                            break
                        
                        # time
                        if self.check_timeout(): break
                    if self.is_timeout: break
                if self.is_timeout: break

        # time
        self.timer_stop()

        ## Warning
        self.end_report()

        # return value
        return guarantee
    


class BottomUpDichotomicDFS(ParallelepipedalSearch):
    """
        * A search algorithm implementing a bottom up dichotomic search in
        the space of guarantees.
        * Expanding one feature at the time.
        * The guarantee passed to the search() method needs to
        have defined the following operations:
            * `expand_dichotomic_ub(i, j)`
            * `expand_dichotomic_lb(i, j)`
            * `up_high_pivot(i, j)`
            * `down_high_pivot(i, j)`
            * `up_low_pivot(i, j)`
            * `down_low_pivot(i, j)`
    """

    def __init__(
                    self,
                    isSAT,
                    max_it = 100,   # max number of iterations
                    timeout = 60,
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, timeout, verbose)

        ## Reporting
        self.msg_prefix = "Bottom-Up Dich. DFS"
        self.prop_name  = "Soundness"

    def search(
            self,
            guarantee: typing.Union[
                parallel.BottomParallelGurantee,
                parallel.BottomDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        self.timer_start()

        # main loop
        # expand *upper bound* with dichotomic search
        for i in range(guarantee.row_dim):
            for j in range(guarantee.column_dim):
                for it in range(self.max_it):
                    ## if dichotomic search converged, break
                    if not guarantee.high_dichotomic_invariant(i, j):
                        self.print_debug(" break due to hich dichotomic invariance!")
                        break


                    ## Refine the explanation
                    self.refinement_success = guarantee.expand_dichotomic_ub(i, j)
                    if not self.refinement_success:
                        self.print_debug(" break due to unsuccessful refinement!")
                        break

                    ## Reporting
                    self.num_it += 1
                    self.progress_message()

                    ## Check convergance
                    self.soundness, _ = self.isSAT(guarantee.get_interval())
                    if self.soundness:
                        self.print_debug(" successful expansion!")
                        succ_pivot_refinement = guarantee.up_high_pivot(i, j)
                        if not succ_pivot_refinement: break
                    
                    else:
                        self.print_debug(" unsuccessful expansion!")
                        succ_pivot_refinement = guarantee.down_high_pivot(i, j)
                        if not succ_pivot_refinement: break
                    
                    # time
                    if self.check_timeout(): break
                if self.is_timeout: break
            if self.is_timeout: break
                


        # expand *lower bound* with dichotomic search
        if not self.is_timeout:
            for i in range(guarantee.row_dim):
                for j in range(guarantee.column_dim):
                    for it in range(self.max_it):
                        ### if dichotomic search converged, break
                        if not guarantee.low_dichotomic_invariant(i, j):
                            self.print_debug(" break due to low dichotomic invariance!")
                            break


                        ## Refine the explanation
                        self.refinement_success = guarantee.expand_dichotomic_lb(i, j)
                        if not self.refinement_success:
                            self.print_debug(" break due to unsuccessful refinement!")
                            break

                        ## Reporting
                        self.num_it += 1
                        self.progress_message()

                        ## Check convergance
                        self.soundness, _ = self.isSAT(guarantee.get_interval())
                        if self.soundness:
                            self.print_debug(" successful expansion!")
                            succ_pivot_refinement = guarantee.down_low_pivot(i, j)
                            if not succ_pivot_refinement: break
                        
                        else:
                            succ_pivot_refinement = guarantee.up_low_pivot(i, j)
                            self.print_debug(" unsuccessful expansion!")
                            if not succ_pivot_refinement: break
                        
                        # time
                        if self.check_timeout(): break
                    if self.is_timeout: break
                if self.is_timeout: break

        
        ## peculiarity of dichotomic search
        if not self.soundness:
            guarantee.make_sound()
            self.soundness = True and not self.is_timeout
        
        # time
        self.timer_stop()

        ## Warning
        self.end_report()

        # return value
        return guarantee


class BottomUpBFS(ParallelepipedalSearch):
    """
        #### Description:
        * A search algorithm implementing a bottom up search in
        the space of guarantees.
        * Expanding one feature at the time.
        * The guarantee passed to the search() method needs to
        have defined the following methods:
            * `expand_ub(i, j)`
            * `expand_lb(i, j)`
            * `revert_expand_ub(i, j)`
            * `revert_expand_lb(i, j)`
    """

    def __init__(
                    self,
                    isSAT,
                    max_it = 100,   # max number of iterations
                    timeout = 60,
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, timeout, verbose)

        ## Reporting
        self.msg_prefix = "Bottom-Up BFS "
        self.prop_name  = "Soundness"

    def search(
            self,
            guarantee: typing.Union[
                parallel.BottomParallelGurantee,
                parallel.BottomDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        self.timer_start()

        ## expand upper bound
        # BFS' queue
        Q = [
                (i, j)  for i in range(guarantee.row_dim)
                        for j in range(guarantee.column_dim)
            ]
        for it in range(self.max_it):
            ## check queue
            if Q == []: break

            (i, j) = Q.pop()
            self.refinement_success = guarantee.expand_ub(i, j)
            if not self.refinement_success: continue

            ## Reporting
            self.num_it += 1
            self.progress_message()

            ## Check convergance
            self.soundness, _ = self.isSAT(guarantee.get_interval())
            if not self.soundness:
                # Since we start with the trivial explanation and expand
                # the explanation will always be sound, until a counter
                # example is provided. If that hapens, we revert to the
                # previous sound explanation.
                self.soundness = True
                guarantee.revert_expand_ub(i, j)
                continue

            else: Q.append((i, j))

            if self.check_timeout(): break
        
        

        ## expand lower bound
        # BFS' queue
        Q = [
                (i, j)  for i in range(guarantee.row_dim)
                        for j in range(guarantee.column_dim)
            ]
        if not self.is_timeout:
            for it in range(self.max_it):
                ## check queue
                if Q == []: break

                (i, j) = Q.pop()
                self.refinement_success = guarantee.expand_lb(i, j)
                if not self.refinement_success: continue

                ## Reporting
                self.num_it += 1
                self.progress_message()

                ## Check convergance
                self.soundness, _ = self.isSAT(guarantee.get_interval())
                if not self.soundness:
                    # Since we start with the trivial explanation and expand
                    # the explanation will always be sound, until a counter
                    # example is provided. If that hapens, we revert to the
                    # previous sound explanation.
                    self.soundness = True
                    guarantee.revert_expand_lb(i, j)
                    continue
                
                else: Q.append((i, j))

                if self.check_timeout(): break



        
        # time
        self.timer_stop()

        ## Warning
        self.end_report()

        # return value
        return guarantee
