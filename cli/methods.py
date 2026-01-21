## Typing
import typing
import numpy as np
import geometry.interval as geom
import verification.nn_verification as nn_verif

## Custom Libraries
import guarantees.parallelepipedal as psg
import guarantees.cyclic as csg

# algo libraries
import algorithms.algorithms as algos
import algorithms.parallelepipedal as palgos
import algorithms.cyclic as calgos
import algorithms.composition as comp





############################
# Parallelepipedal Methods #
############################

## Bottom-Up Methods ##

def init_bottom_up_linear_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[psg.ParallelepipedalGuarantee, algos.SearchAlgorithm]:

    return  psg.BottomDistParallelGurantee(x_star, c_star, rad, delta, domain),\
            palgos.BottomUpLinearDFS(isSAT, max_it, timeout, verbose)


def init_bottom_up_dichotomic_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[psg.ParallelepipedalGuarantee, algos.SearchAlgorithm]:

    return  psg.BottomDistParallelGurantee(x_star, c_star, rad, delta, domain),\
            palgos.BottomUpDichotomicDFS(isSAT, max_it, timeout, verbose)


def init_bottom_up_bfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[psg.ParallelepipedalGuarantee, algos.SearchAlgorithm]:

    return  psg.BottomDistParallelGurantee(x_star, c_star, rad, delta, domain),\
            palgos.BottomUpBFS(isSAT, max_it, timeout, verbose)



## Top-Down Methods ##

def init_top_down(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[psg.ParallelepipedalGuarantee, algos.SearchAlgorithm]:

    return  psg.TopDistParallelGurantee(x_star, c_star, rad, delta, domain),\
            palgos.TopDownSearch(isSAT, max_it, timeout, verbose)



## Supported Methods Compositions (Between Parallel Search Algos) ##

###############################################################################
# ParallelAlgoComposition = ParallelepipedalSearch + ParallelepipedalSearch
#
# The suppoerted compositions are:
#   1. Top-Down + Bottom-Up Linear DFS
#   2. Top-Down + Bottom-Up Dichotomic DFS
#   2. Top-Down + Bottom-Up BFS
#
#   Recommended: Top-Down + Bottom-Up Dichotomic DFS
#
# Notes:
# ----------------------------------------------------------------------------
# * The above composition operation "+" is *not* commutative! In other
#   words, the composition of the form <Bottom-Up Search> + <Top-Down Search>
#   is *not* supported or theoretically well defined.
# 
# * On the other hand, we can compose two bottom up algorithms, i.e.
# compositions of the form <Bottom-Up Search> + <Bottom-Up Search> are well
# defined. On the other hand, they are theoretically *redundant*:
#
#   1.  Any of the bottom-up algorithms terminates after achieving *maximality*.
#   2.  Thus, a bottom-up algorithm will *not* even begin, when an already
#       maximal guarantee is passed to the algorithm as input.
#   3.  Hence it holds:
#
#       <Bottom-Up Algo 1> + <Bottom-Up Algo 2> = <Bottom-Up Algo 1>
#
#       Therefore, supporting these compositions is redundant.
###############################################################################


def init_td_n_bu_l_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[psg.ParallelepipedalGuarantee, algos.SearchAlgorithm]:

    guarantee  = psg.TopDistParallelGurantee(x_star, c_star, rad, delta, domain)
    algo1      = palgos.TopDownSearch(isSAT, max_it, timeout, verbose)
    algo2      = palgos.BottomUpLinearDFS(isSAT, int(rad/delta), timeout, verbose)
    algo       = comp.ParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

    return  guarantee, algo



def init_td_n_bu_d_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[psg.ParallelepipedalGuarantee, algos.SearchAlgorithm]:

    guarantee  = psg.TopDistParallelGurantee(x_star, c_star, rad, delta, domain)
    algo1      = palgos.TopDownSearch(isSAT, max_it, timeout, verbose)
    algo2      = palgos.BottomUpDichotomicDFS(isSAT, int(rad/delta), timeout, verbose)
    algo       = comp.ParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

    return  guarantee, algo


def init_td_n_bu_bfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[psg.ParallelepipedalGuarantee, algos.SearchAlgorithm]:

    guarantee  = psg.TopDistParallelGurantee(x_star, c_star, rad, delta, domain)
    algo1      = palgos.TopDownSearch(isSAT, max_it, timeout, verbose)
    algo2      = palgos.BottomUpBFS(isSAT, int(rad/delta), timeout, verbose)
    algo       = comp.ParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

    return  guarantee, algo



##################
# Cyclic Methods #
##################


## Bottom-Up Methods ##

def init_cyclic_bottom_up_linear(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

    return  csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain),\
            calgos.BottomUpLinearSearch(isSAT, max_it, timeout, verbose)


def init_cyclic_bottom_up_dichotomic(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

    return  csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain),\
            calgos.BottomUpDichotomicSearch(isSAT, max_it, timeout, verbose)



## Top-Down Methods ##

def init_cyclic_top_down(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

    return  csg.TopCyclicGuarantee(x_star, c_star, rad, delta, domain),\
            calgos.TopDownSearch(isSAT, max_it, timeout, verbose)



## Supported Methods Compositions (Between Cyclic and Parallelepipedal Search Algos) ##


###############################################################################
# CyclicParallelAlgoComposition = CycliclSearch + ParallelepipedalSearch
#
# The suppoerted compositions are:
#
#   1. Cyclic Bottom-Up Linear Search + Bottom-Up Linear DFS
#   2. Cyclic Bottom-Up Linear Search + Bottom-Up Dichotomic DFS
#   3. Cyclic Bottom-Up Linear Search + Bottom-Up BFS
#
#   4. Cyclic Bottom-Up Dichotomic Search + Bottom-Up Linear DFS
#   5. Cyclic Bottom-Up Dichotomic Search + Bottom-Up Dichotomic DFS
#   6. Cyclic Bottom-Up Dichotomic Search + Bottom-Up BFS
#
#   7. Cyclic Top-Down + Bottom-Up Linear DFS
#   8. Cyclic Top-Down + Bottom-Up Dichotomic DFS
#   9. Cyclic Top-Down + Bottom-Up BFS
#
# Notes:
# ----------------------------------------------------------------------------
# * Again the composition operation "+" is *not* commutative.
# * It holds:
#   1.  <Parallel Search> + <Cyclic Search> = <Parallel Search>
#   2.  <Cyclic Search Algo 1> + <Cyclic Search Algo 2> = <Cyclic Search Algo 1>
#   3.  <Cyclic Search> + <Top-Down Parallel Search> = <Cyclic Search>
###############################################################################


# Cyclic Bottom-Up Linear + <Parallelepipedal Bottom-Up Search>

def init_cbu_l_n_bu_l_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.BottomUpLinearSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpLinearDFS(isSAT, int(rad/delta), timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo


def init_cbu_l_n_bu_d_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.BottomUpLinearSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpDichotomicDFS(isSAT, int(rad/delta),  timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo


def init_cbu_l_n_bu_bfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.BottomUpLinearSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpBFS(isSAT, int(rad/delta), timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo



# Cyclic Bottom-Up Dichotomic + <Parallelepipedal Bottom-Up Search>


def init_cbu_d_n_bu_l_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.BottomUpDichotomicSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpLinearDFS(isSAT, int(rad/delta), timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo


def init_cbu_d_n_bu_d_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.BottomUpDichotomicSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpDichotomicDFS(isSAT, int(rad/delta), timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo


def init_cbu_d_n_bu_bfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.BottomUpDichotomicSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpBFS(isSAT, int(rad/delta), timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo


# Cyclic Top-Down + <Parallelepipedal Bottom-Up Search>

def init_ctd_n_bu_l_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.TopDownSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpLinearDFS(isSAT, int(rad/delta), timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo


def init_ctd_n_bu_d_dfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.TopDownSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpDichotomicDFS(isSAT, int(rad/delta), timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo


def init_ctd_n_bu_bfs(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

        guarantee  = csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain)
        algo1      = calgos.TopDownSearch(isSAT, max_it, timeout, verbose)
        algo2      = palgos.BottomUpBFS(isSAT, int(rad/delta), timeout, verbose)
        algo       = comp.CyclicParallelAlgoComposition(algo1, algo2, isSAT, max_it, verbose)

        return  guarantee, algo


## Methods for Complete Approximations
def init_complete_bu(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[psg.ParallelepipedalGuarantee, algos.SearchAlgorithm]:

    return  psg.BottomDistParallelGurantee(x_star, c_star, rad, delta, domain),\
            palgos.CompleteBottomUpSearch(isSAT, max_it, timeout, verbose)


def init_complete_c_d_bu(
        x_star:     np.ndarray,
        c_star:     int,
        rad:        float,
        delta:      float,
        domain:     geom.Interval,
        isSAT:      nn_verif.NNVerification,
        max_it:     int,
        timeout:    int,
        verbose:    bool
    ) -> typing.Tuple[csg.CyclicGuarantee, algos.SearchAlgorithm]:

    return  csg.BottomCyclicGuarantee(x_star, c_star, rad, delta, domain),\
            calgos.CompleteBottomUpDichotomicSearch(isSAT, max_it, timeout, verbose)


###############
# Methods Ids #
###############


# Parallelepipedal Methods
bottom_up_linear_dfs        = 0
bottom_up_dichotomic_dfs    = 1
bottom_up_bfs               = 2
top_down                    = 3

# Cyclic Methods
cyclic_bottom_up_linear     = 4
cyclic_bottom_up_dichotomic = 5
cyclic_top_down             = 6

# Parallel + Parallel Compositions
td_n_bu_l_dfs               = 7
td_n_bu_d_dfs               = 8
td_n_bu_bfs                 = 9

# Cyclic + Parallel Compositions
cbu_l_n_bu_l_dfs            = 10
cbu_l_n_bu_d_dfs            = 11
cbu_l_n_bu_bfs              = 12

cbu_d_n_bu_l_dfs            = 13
cbu_d_n_bu_d_dfs            = 14
cbu_d_n_bu_bfs              = 15

ctd_n_bu_l_dfs              = 16
ctd_n_bu_d_dfs              = 17
ctd_n_bu_bfs                = 18

## Algorithms for Complete Approximations
complete_bu                 = 19
complete_c_d_bu             = 20

## Types, types, types.. types everywhere
GuaranteeUnion_t    = typing.Union[
                                csg.CyclicGuarantee,
                                psg.ParallelepipedalGuarantee
                        ]
InitMethod_t        = typing.Callable[
                                [
                                      np.ndarray,
                                      int,
                                      float,
                                      float,
                                      geom.Interval,
                                      nn_verif.NNVerification,
                                      int,
                                      int,
                                      bool
                                ],
                                typing.Tuple[
                                      GuaranteeUnion_t,
                                      algos.SearchAlgorithm
                                ]
                        ]

init_method: typing.Dict[int, InitMethod_t] = {
    # Parallelepipedal Methods
    bottom_up_linear_dfs:           init_bottom_up_linear_dfs,
    bottom_up_dichotomic_dfs:       init_bottom_up_dichotomic_dfs,
    bottom_up_bfs:                  init_bottom_up_bfs,
    top_down:                       init_top_down,

    # Cyclic Methods
    cyclic_bottom_up_linear:        init_cyclic_bottom_up_linear,
    cyclic_bottom_up_dichotomic:    init_cyclic_bottom_up_dichotomic,
    cyclic_top_down:                init_cyclic_top_down,

    # Parallel + Parallel Compositions
    td_n_bu_l_dfs:                  init_td_n_bu_l_dfs,
    td_n_bu_d_dfs:                  init_td_n_bu_d_dfs,
    td_n_bu_bfs:                    init_td_n_bu_bfs,

    # Cyclic + Parallel Compositions
    cbu_l_n_bu_l_dfs:               init_cbu_l_n_bu_l_dfs,
    cbu_l_n_bu_d_dfs:               init_cbu_l_n_bu_d_dfs, 
    cbu_l_n_bu_bfs:                 init_cbu_l_n_bu_bfs, 

    cbu_d_n_bu_l_dfs:               init_cbu_d_n_bu_l_dfs,
    cbu_d_n_bu_d_dfs:               init_cbu_d_n_bu_d_dfs,
    cbu_d_n_bu_bfs:                 init_cbu_d_n_bu_bfs,

    ctd_n_bu_l_dfs:                 init_ctd_n_bu_l_dfs,
    ctd_n_bu_d_dfs:                 init_ctd_n_bu_d_dfs,
    ctd_n_bu_bfs:                   init_ctd_n_bu_bfs,

    ## Algorithms for Complete Approximations
    complete_bu:                    init_complete_bu,
    complete_c_d_bu:                init_complete_c_d_bu
}