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
            verbose:    bool = False
        ):
        super().__init__(isSAT, max_it, verbose)
    
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
                    max_it = 1000,   # max number of iterations
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Top-Down " + self.msg_prefix
    

    def search(
            self,
            guarantee: typing.Union[
                parallel.TopParallelGuarantee,
                parallel.TopDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
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
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Complete Bottom-Up " + self.msg_prefix

    ###########
    # Reports #
    ###########
    def end_report(self):
        if self.verbose:
            print("\n#",self.msg_prefix, "End Report")
            print(f"{'Completeness:':<20}"          + str(self.completeness))
            print(f"{'Refinement Success:':<20}"    + str(self.refinement_success))
            print(f"{'No. of Iterations:':<20}"     + str(self.num_it))
            print(f"{'Total Time:':<20}"            + str(round(self.total_time, 2)) + " (secs)")
            print(f"{'Verif. Time:':<20}"           + str(round(self.isSAT.get_total_time(), 2)) + " (secs)")
            print(f"{'Verif. Num. Calls:':<20}"     + str(self.isSAT.get_num_calls()))
            print(f"{'Verif. Avg Time:':<20}"       + str(round(self.isSAT.get_avg_time(), 2)) + " (secs)")
            print(f"{'Verif. Time Perc.:':<20}"     + str(round(self.isSAT.get_total_time() / self.total_time, 4) * 100) + "%")

    def search(
            self,
            guarantee: typing.Union[
                parallel.BottomParallelGurantee,
                parallel.BottomDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        tic = time.time()

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
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Bottom-Up Lin. DFS"
    

    def search(
            self,
            guarantee: typing.Union[
                parallel.BottomParallelGurantee,
                parallel.BottomDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        tic = time.time()

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


        # expand *lower bound* with linear search
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
        toc = time.time()
        self.total_time = toc - tic

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
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Bottom-Up Dich. DFS"

    def search(
            self,
            guarantee: typing.Union[
                parallel.BottomParallelGurantee,
                parallel.BottomDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        tic = time.time()

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
                


        # expand *lower bound* with dichotomic search
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
                    verbose = False
                ):
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Bottom-Up BFS "

    def search(
            self,
            guarantee: typing.Union[
                parallel.BottomParallelGurantee,
                parallel.BottomDistParallelGurantee
            ]
        ) -> parallel.ParallelepipedalGuarantee:
        # time
        tic = time.time()

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
        
        

        ## expand lower bound
        # BFS' queue
        Q = [
                (i, j)  for i in range(guarantee.row_dim)
                        for j in range(guarantee.column_dim)
            ]
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



        
        # time
        toc = time.time()
        self.total_time = toc - tic

        ## Warning
        self.end_report()

        # return value
        return guarantee
