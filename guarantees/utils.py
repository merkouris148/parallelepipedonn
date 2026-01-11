import guarantees.parallelepipedal as parallel
import guarantees.cyclic as cyclic

import copy


def cyclic2parallel(
        cyclic_guarantee: cyclic.CyclicGuarantee
    ) -> parallel.ParallelepipedalGuarantee:
    """
        #### Description:
        Constructs an instance of the ParallelepipedalGurantee
        class from a CyclicGuarantee.

        #### Notes:
        This operation is only possible (mathematically) for
        `\ell_\infty` circles. For other norms it is not well
        defined.
    """
    
    int_cyclic = cyclic_guarantee.get_interval()
    int_cyclic.intersect(cyclic_guarantee.domain)

    # We use DistParallelGuarantees since it is a *generalization*
    # of ParallelGuarantees.
    parallel_dist_guarantee = parallel.BottomDistParallelGurantee(
        cyclic_guarantee.center,
        cyclic_guarantee.c_star,
        cyclic_guarantee.distance_restriction,
        cyclic_guarantee.delta,
        cyclic_guarantee.domain
    )

    lb_updated, ub_updated = parallel_dist_guarantee.set_bounds(int_cyclic.lb, int_cyclic.ub)
    assert lb_updated and ub_updated
    
    return copy.deepcopy(parallel_dist_guarantee)


# The following can be defined in the following ways:
#   a) the inscribed l_\infty circle
#   b) the circumscribed l_\infty circle
#
# Neither is very usefull for the MNIST dataset!
#
# def parallel2cyclic(
#         parallel_guarantee: parallel.ParallelepipedalGuarantee
#     ) -> cyclic.CyclicGuarantee:

#     #pass
#     raise NotImplemented