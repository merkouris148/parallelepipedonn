from algorithms.algorithms import SearchAlgorithm
from algorithms.cyclic import CyclicSearch
from algorithms.parallelepipedal import ParallelepipedalSearch

# libraries for typing
import typing
from verification.nn_verification import NNVerification
import guarantees.cyclic as cyclic
import guarantees.parallelepipedal as parallel
import guarantees.utils as utils

# python libraries
from copy import copy

class AlgoComposition(SearchAlgorithm):
    def __init__(
                    self,
                    algo1:      SearchAlgorithm,
                    algo2:      SearchAlgorithm,
                    isSAT:      NNVerification,
                    max_it:     int = 100,   # max number of iterations
                    verbose:    bool = False
                ) -> None:
        
        super().__init__(isSAT, max_it, verbose)

        ## Reporting
        self.msg_prefix = "Algo. Comp.:" + " " + algo1.msg_prefix + "+" + " " + algo2.msg_prefix

        # algorithms
        self.algo1 = algo1
        self.algo2 = algo2

        # tinkering msg prefixes
        self.algo1.msg_prefix = self.algo1.msg_prefix
        self.algo2.msg_prefix = self.algo2.msg_prefix
    

    def algo1_prep(self,
            guarantee: typing.Union[
                cyclic.CyclicGuarantee,
                parallel.ParallelepipedalGuarantee
            ]
        ) -> typing.Union[
                cyclic.CyclicGuarantee,
                parallel.ParallelepipedalGuarantee
        ]:

        raise NotImplementedError


    def algo2_prep(self,
            guarantee: typing.Union[
                cyclic.CyclicGuarantee,
                parallel.ParallelepipedalGuarantee
            ]
        ) -> typing.Union[
                cyclic.CyclicGuarantee,
                parallel.ParallelepipedalGuarantee
        ]:

        raise NotImplementedError

    
    def type_conversion(self,
            guarantee: typing.Union[
                cyclic.CyclicGuarantee,
                parallel.ParallelepipedalGuarantee
            ]
        ) -> typing.Union[
                cyclic.CyclicGuarantee,
                parallel.ParallelepipedalGuarantee
        ]:

        raise NotImplementedError


    def search(
            self,
            guarantee: typing.Union[
                cyclic.CyclicGuarantee,
                parallel.ParallelepipedalGuarantee
            ]
        ) -> typing.Union[
                cyclic.CyclicGuarantee,
                parallel.ParallelepipedalGuarantee
        ]:
        
        guarantee = self.algo1_prep(guarantee)

        self.print("\nApplying algo. 1: " + self.algo1.msg_prefix + "\n")
        guarantee = self.algo1.search(copy(guarantee))

        ## preparing the explanation for the bu-d-dfs algorithm
        #guarantee.update_pivots()

        guarantee = self.type_conversion(guarantee)
        guarantee = self.algo2_prep(guarantee)

        self.print("\nApplying algo. 2: " + self.algo2.msg_prefix + "\n")
        guarantee = self.algo2.search(copy(guarantee))

        self.total_time = self.algo1.total_time +   self.algo2.total_time
        self.soundness  = self.algo1.soundness  and self.algo2.soundness
        self.num_it     = self.algo1.num_it     +   self.algo2.num_it

        ## composition's end report
        self.end_report()



        return copy(guarantee)


class ParallelAlgoComposition(AlgoComposition):

    def __init__(
            self,
            algo1:      ParallelepipedalSearch,
            algo2:      ParallelepipedalSearch,
            isSAT:      NNVerification,
            max_it:     int = 100,
            verbose:    bool = False
        ) -> None:
        
        super().__init__(algo1, algo2, isSAT, max_it, verbose)
    

    def algo1_prep(
            self,
            guarantee: parallel.ParallelepipedalGuarantee
        ) -> parallel.ParallelepipedalGuarantee:
        """
            #### Description:
            Identical function.
        """

        return guarantee


    def algo2_prep(
            self,
            guarantee: parallel.ParallelepipedalGuarantee
        ) -> parallel.ParallelepipedalGuarantee:
        """
            #### Description:
            Identical function.
        """

        guarantee.update_pivots()
        return guarantee


    ## type conversion (identical function)
    def type_conversion(self,
            guarantee: parallel.ParallelepipedalGuarantee
        ) -> parallel.ParallelepipedalGuarantee:
        """
            #### Description:
            Identical function.
        """

        return guarantee


class CyclicAlgoComposition(AlgoComposition):

    def __init__(
            self,
            algo1:      CyclicSearch,
            algo2:      CyclicSearch,
            isSAT:      NNVerification,
            max_it:     int = 100,
            verbose:    bool = False
        ) -> None:
        super().__init__(algo1, algo2, isSAT, max_it, verbose)
    

    def algo1_prep(
            self,
            guarantee: cyclic.CyclicGuarantee
        ) -> cyclic.CyclicGuarantee:
        """
            #### Description:
            Identical function.
        """

        return guarantee


    def algo2_prep(
            self,
            guarantee: cyclic.CyclicGuarantee
        ) -> cyclic.CyclicGuarantee:
        """
            #### Description:
            Identical function.
        """

        return guarantee

    ## type conversion (identical function)
    def type_conversion(self,
            guarantee: cyclic.CyclicGuarantee
        ) -> cyclic.CyclicGuarantee:
        """
            #### Description:
            Identical function.
        """

        return guarantee


class CyclicParallelAlgoComposition(AlgoComposition):

    def __init__(
            self,
            algo1:      CyclicSearch,
            algo2:      ParallelepipedalSearch,
            isSAT:      NNVerification,
            max_it:     int = 100,
            verbose:    bool = False
        ) -> None:
        super().__init__(algo1, algo2, isSAT, max_it, verbose)
    
    
    def algo1_prep(
            self,
            guarantee: cyclic.CyclicGuarantee
        ) -> cyclic.CyclicGuarantee:
        """
            #### Description:
            Identical function.
        """

        return guarantee


    def algo2_prep(
            self,
            guarantee: parallel.ParallelepipedalGuarantee
        ) -> parallel.ParallelepipedalGuarantee:
        """
            #### Description:
            Updating pivots before beginning the second algorithm.
        """

        ## When changing bounds we NEED to update the pivots!
        guarantee.update_pivots()
        return guarantee

    ## type conversion (identical function)
    def type_conversion(self,
            guarantee: cyclic.CyclicGuarantee
        ) -> parallel.ParallelepipedalGuarantee:
        """
            #### Description:
            Converting a cyclic guarantee to a parallelepipedal.
        """

        return utils.cyclic2parallel(guarantee)