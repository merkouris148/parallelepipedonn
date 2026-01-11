"""
    Put a docstring here blah blah
"""

#############
# Libraries #
#############

## Python Libraries
import os

## 3rd Party Libraries
import numpy as np
import matplotlib.image as plt_im

## Custom
import cli.methods as methods
import cli.verifiers as verifiers
import cli.error_handling as errors
import cli.args as args
#import config

import verification.marabou as marabou_verif
import guarantees.parallelepipedal as psg
import guarantees.cyclic as csg
import geometry.interval as geom



# overwrite given TF's c_star by Marabou's estimation
# (seems not usefull in practice)
# Marabou's representatrion is a good approximation
# of the TF network
overwrite_given_prediction = False



####################
# Experiment Class #
####################
class Application:
    """
        #### Description:
        The class encoding the ParallelepipedoNN application. Essentially
        configure a *Stability Guarantee* instance. Namely, computing the
        following pseudo-lambda term:
        ```
            (λ alg. λ orc. λ guar. alg orc guar)
                (λ m. λ r. λ δ. λ ver. search_algo m r δ ver)   # Construct the algorithm
                (λ onnx. λ c*. oracle onnx c*)                  # Construct the oracle
                (λ x*. λ c*. λ dom. guarantee x* c* dom)        # Construct the initial guarantee
        ```
    """

    def __init__(
            self,

            ## NN parameters
            x_star_path:    str,    # single data vector
            c_star:         int,    # nn predictions
            onnx_path:      str,    # path to the nn
            output_path:    str,

            ## Guarantee parameters
            verifier:       int,
            method:         int,    # specified algorithm for guarantees
            max_it:         int,    # max number of iterations
            rad:            float,  # radius of distance restriction
            delta:          float,  # percision constant
            
            ## Domain
            # we define a domain of the form
            #   [dom_lb * 1, dom_ub * 1]
            dom_lb:         float,  # domains scalar lb
            dom_ub:         float,  # domains scalar ub

            ## Verbose
            verbose:        bool,

            ## Files
            # the prefix of the path to store the lb, ub csvs
            delimeter:      str = " ",

            ## Bounds from files
            lb_path:        str = "",
            ub_path:        str = ""
        ):

        ####################
        # Check Parameters #
        ####################
        # NN parameters
        assert os.path.isfile(onnx_path), onnx_path
        assert os.path.isfile(x_star_path), x_star_path

        # Guarantee parameters
        assert max_it   > 0
        assert delta    > 0
        assert rad      > delta


        #########################
        # Initialize parameters #
        #########################
        ## Save parameters
        self.x_star_path    = x_star_path
        self.lb_path        = lb_path
        self.ub_path        = ub_path
        self.onnx_path      = onnx_path
        self.output_path    = output_path

        ## Guarantee instance
        self.x_star = np.genfromtxt(x_star_path, delimiter=delimeter)
        self.c_star = c_star

        ## Algo parameters
        self.rad        = rad
        self.delta      = delta
        self.max_it     = max_it
        self.verbose    = verbose

        ## Domain
        self.dom_lb = dom_lb
        self.dom_ub = dom_ub
        self.domain = geom.Interval(
            self.dom_lb * np.ones(self.x_star.shape),
            self.dom_ub * np.ones(self.x_star.shape)
        )



        #########################
        # Initialize the Oracle #
        #########################
        self.isSAT = verifiers.init_method[verifier](
            self.c_star,
            self.onnx_path,
            self.domain
        )

        if not self.check_class_oracle_consistency() and overwrite_given_prediction:
            oracle_prediction = self.isSAT.predict_argmax(self.x_star)[0]
            self.do_overwrite_given_prediction(oracle_prediction, onnx_path)




        ############################
        # Initialize the Algorithm #
        ############################
        self.guarantee  = None  # current explanation
        self.algo       = None  # algorithm

        if method in args.algo_args.keys():
            self.guarantee, \
            self.algo       \
            =               \
            methods.init_method[method](
                    self.x_star,
                    self.c_star,
                    self.rad,
                    self.delta,
                    self.domain,
                    self.isSAT,
                    self.max_it,
                    self.verbose
            )
        else: errors.print_error_message(errors.error_unknown_method)

        
        ##########################################
        # Initialize Bounds from File (if given) #
        ##########################################
        if self.lb_path != "" or self.ub_path != "":
            self.load_bounds(self.lb_path, self.ub_path)
        
        
        ## State
        self.done = False
    

    ####################
    # Input Operations #
    ####################

    def check_class_oracle_consistency(self) -> bool:
        """
                #### Description:
                Check if `c* == κ(x*)`.
            """
        oracle_prediction = self.isSAT.predict_argmax(self.x_star)[0]
        
        ## check that indeed k(x_star) = c_star
        if self.c_star != oracle_prediction:

            warning_message = "Oracle predicts", oracle_prediction, "but", self.c_star, "is given"
            errors.print_warning_message(errors.warning_class_oracle_inconcistency, warning_message)
            
        
        return self.c_star == oracle_prediction
    

    def do_overwrite_given_prediction(self, new_prediction: int, onnx_path: str):
            """
                #### Description:
                Overwrite given prediction `c*` and replace it with the oracles prediction
                for `x*`. Also re-initiallize the oracle (for different `c*` the constrints
                need to be reset).
            """
            
            errors.print_warning_message(errors.warning_overwrite_given_class)

            self.c_star = new_prediction
            self.isSAT  = marabou_verif.MarabouVerification(self.c_star, onnx_path)



    def load_bounds(self, lb_path: str, ub_path: str):
        """
            #### Description:
            Loading `lb`, `ub` from file. This operation is supported *only* for
            parallelepipedal guarantees.
        """
        if not isinstance(self.guarantee, psg.ParallelepipedalGuarantee):
            errors.print_error_message(
                errors.error_bounds_from_file_non_parallel
            )

        ## Check if lb path exists
        new_lb = None
        if lb_path != "":
            assert os.path.isfile(lb_path)
            new_lb = np.genfromtxt(lb_path, delimiter=" ")
        
        ## Check if ub path exists
        new_ub = None
        if ub_path != "":
            assert os.path.isfile(ub_path)
            new_ub = np.genfromtxt(ub_path, delimiter=" ")

        ## Set the given bounds
        self.guarantee.set_bounds(new_lb, new_ub)



    ####################################
    # Apply Algorithm to the Guarantee #
    ####################################

    def apply(self):
        """
            #### Description:
            Applying the algorithm to the guarantee.
            I.e. compute: `(λx. serach_algo x) guarantee`.
        """

        # assert to not redo the same experiment twice
        assert self.done == False
        self.guarantee = self.algo.search(self.guarantee)
        self.done = True
    

    ###########
    # Results #
    ###########
    def print_input(self):
        print("\n# I/O Info")
        print("-----------")
        print(f"{'Input:':<23}"                 + self.x_star_path)
        print(f"{'ONNX Descr.:':<23}"           + self.onnx_path)
        print(f"{'Output Path Pfx:':<23}"       + self.output_path)
        print(f"{'Low. Bound from File:':<23}"  + self.lb_path)
        print(f"{'Up. Bound from File:':<23}"   + self.ub_path)
        print("\n")

    def print_setup(self):
        print("\n# Setup")
        print("--------")
        print(f"{'Method:':<22}"                + self.algo.msg_prefix)
        print(f"{'Max. It.:':<22}"              + str(self.algo.max_it))
        if isinstance(self.guarantee, csg.CyclicGuarantee):
            print(f"{'Radius Dist. Restr.:':<22}"   + str(self.guarantee.distance_restriction))
        else:
            print(f"{'Radius Dist. Restr.:':<22}"   + str(self.guarantee.radius))
        print(f"{'Delta:':<22}"                 + str(self.guarantee.delta))
        print("-" * 60 + "\n")

    def print_results(self):
        print("-" * 60 + "\n")
        print("\n# Results")
        print("----------")
        print(f"{'Num. It.:':<20}"          + str(self.algo.num_it))
        print(f"{'Time:':<20}"              + str(round(self.algo.total_time, 2)) + " (secs)")
        print(f"{'Comp.:':<20}"             + str(self.guarantee.calc_complexity()))
        min_edge_len = None
        if isinstance(self.guarantee, csg.CyclicGuarantee):
            interval = self.guarantee.get_interval()
            interval.intersect(self.domain)
            min_edge_len = interval.min_edge_length()
        else:
            min_edge_len = self.guarantee.min_edge_length()
        print(f"{'Min. Edge Length:':<20}"      + str(round(min_edge_len, 4)))
        print(f"{'Verif. Time:':<20}"           + str(round(self.isSAT.get_total_time(), 2)) + " (secs)")
        print(f"{'Verif. Num. Calls:':<20}"     + str(self.isSAT.get_num_calls()))
        print(f"{'Verif. Avg Time:':<20}"       + str(round(self.isSAT.get_avg_time(), 2)) + " (secs)")
        print(f"{'Verif. Time Perc.:':<20}"     + str(round(self.isSAT.get_total_time() / self.algo.total_time, 4) * 100) + "%")
    

    def print_simple_results(self):
        min_edge_len = None
        if isinstance(self.guarantee, csg.CyclicGuarantee):
            interval = self.guarantee.get_interval()
            interval.intersect(self.domain)
            min_edge_len = interval.min_edge_length()
        else:
            min_edge_len = self.guarantee.min_edge_length()

        simple_res =    str(self.algo.num_it)                                           + " " +\
                        str(round(self.algo.total_time, 2))                             + " " +\
                        str(self.guarantee.calc_complexity())                           + " " +\
                        str(round(min_edge_len, 4))                                     + " " +\
                        str(round(self.isSAT.get_total_time(), 2))                      + " " +\
                        str(self.isSAT.get_num_calls())                                 
        
        print(simple_res)


    #####################
    # Output Operations #
    #####################

    def save_bounds(self, overwrite = False) -> None:
        """
            #### Description:
            Saving the bounds to `<output_prefix>_lb.csv`
            `<output_prefix>_ub.csv`, *if the files do not exist*.
        """

        assert self.done == True

        guarantee_interval = self.guarantee.get_interval()

        ub_output_path = self.output_path + "_ub.csv"
        if not os.path.isfile(ub_output_path) or overwrite: np.savetxt(ub_output_path, guarantee_interval.ub)

        lb_output_path = self.output_path + "_lb.csv"
        if not os.path.isfile(lb_output_path) or overwrite: np.savetxt(lb_output_path, guarantee_interval.lb)

    
    def image_save_bounds(self, overwrite = False) -> None:
        """
            #### Description:
            Saving the images to `<output_prefix>_lb.png`
            `<output_prefix>_ub.png`, *if the files do not exist*.
        """

        assert self.done == True

        guarantee_interval = self.guarantee.get_interval()

        ub_output_path = self.output_path + "_ub.png"
        if not os.path.isfile(ub_output_path) or overwrite:
            plt_im.imsave(ub_output_path, guarantee_interval.ub, cmap="Greys", vmin=self.dom_lb, vmax=self.dom_ub)

        lb_output_path = self.output_path + "_lb.png"
        if not os.path.isfile(lb_output_path) or overwrite:
            plt_im.imsave(lb_output_path, guarantee_interval.lb, cmap="Greys", vmin=self.dom_lb, vmax=self.dom_ub)