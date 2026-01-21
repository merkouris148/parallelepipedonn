## Libraries for Typing
import typing
import sys
import verification.nn_verification as nn_verif

# time
import time

## Mesage prefix
msg_prefix = ""

#########
# Debug #
#########
DEBUG = False


class SearchAlgorithm:
    """
        #### Description:
        The *top class* of a search algorithm.
        * Each subclass implements a different search strategy.
        * Each subclass needs to define a search() method, that
        takes as input an explanation and returns an explanation
        * The statistics properties of this class encode the 
        properties of the explanation.
    """
    def __init__(
                    self,
                    isSAT:      nn_verif.NNVerification,
                    max_it:     int = 1000,     # max number of iterations
                    timeout:    int = 60,       # timeout in minutes
                    verbose:    int = False
                ) -> None:

        assert max_it   > 0
        assert timeout  > 0

        ## Parameters
        self.isSAT      = isSAT
        self.max_it     = max_it
        self.verbose    = verbose
        self.prop_name  = ""        # The property needed to be verified
        
        ## Statistics
        self.soundness          = False
        self.completeness       = False
        self.num_it             = 0         # we want num_it to count
                                            # the number of isSAT oracle calls
        self.refinement_success = True

        ## Time
        self.timeout    = timeout
        self.is_timeout = False     # did the search time out?
        self.tic        = 0
        self.toc        = 0
        self.total_time = 0
        

        ## Reports
        self.msg_prefix = msg_prefix
    
    ## Mutators
    def reset_algo(self):
        self.soundness          = False
        self.num_it             = 0
        self.refinement_success = True
        self.total_time         = 0

    ## Accessors
    def get_statistics(self) -> typing.List[typing.Union[bool, int, float]]:
        return [
            self.soundness,
            self.num_it,
            self.refinement_success,
            self.total_time
        ]
    
    ## Time
    def timer_start(self) -> None:
        self.tic = time.time()
    
    def timer_stop(self) -> None:
        self.toc        = time.time()
        self.total_time = self.toc - self.tic

    def check_timeout(self) -> bool:
        toc             = time.time()
        elapsed_time    = (toc - self.tic) / 60
        if elapsed_time > self.timeout:
            self.is_timeout = True
            self.timeout_message()
            return True
        
        return False

    ## Reporting
    def progress_message(self):
        if self.num_it % 10 == 0 and self.verbose:
            print(self.msg_prefix, "Iteration ", self.num_it)
    
    def timeout_message(self):
        if self.verbose: print(self.msg_prefix, "Time out!")

    def print(self, msg):
        if self.verbose: print(self.msg_prefix, msg)
    
    def print_debug(self, msg):
        if DEBUG:
            print(
                self.msg_prefix, ":", "it.:", self.num_it, msg,
                file=sys.stderr
            )

    def end_report(self):
        timeout_str = "Yes" if self.is_timeout else "No"

        if self.verbose:
            print("\n#",self.msg_prefix, "End Report")
            print(f"{self.prop_name:<20}"           + str(self.soundness))
            print(f"{'Refinement Success:':<20}"    + str(self.refinement_success))
            print(f"{'No. of Iterations:':<20}"     + str(self.num_it))
            print(f"{'Timeout:':<20}"               + str(self.timeout) + "(mins)")
            print(f"{'Search Time Out:':<20}"       + str(self.is_timeout))
            print(f"{'Total Time:':<20}"            + str(round(self.total_time, 2)) + " (secs)")
            print("-" * 60)
            print(f"{'Verif. Time:':<20}"           + str(round(self.isSAT.get_total_time(), 2)) + " (secs)")
            print(f"{'Verif. Num. Calls:':<20}"     + str(self.isSAT.get_num_calls()))
            print(f"{'Verif. Timeouts:':<20}"       + str(self.isSAT.get_timeouts()))
            print(f"{'Verif. Avg Time:':<20}"       + str(round(self.isSAT.get_avg_time(), 2)) + " (secs)")
            print(f"{'Verif. Time Perc.:':<20}"     + str(round(self.isSAT.get_total_time() / self.total_time, 4) * 100) + "%")