###################
# NN Varification #
###################
class NNVerification:
    ## Constructor
    def __init__(self, c_star, model_description):
        # Parameters
        self.c_star             = c_star
        self.model_description  = model_description

        # Statistics
        self.total_time     = 0
        self.num_calls      = 0
        self.num_timeouts   = 0
    
    ## Accessors
    def get_avg_time(self) -> float:
        return self.total_time / self.num_calls
    
    def get_total_time(self) -> float:
        return self.total_time

    def get_num_calls(self) -> int:
        return self.num_calls
    
    def get_timeouts(self) -> int:
        return self.num_timeouts

    ## Mutators
    def set_statistics(self, call_time:float):
        assert call_time >= 0

        self.num_calls  += 1
        self.total_time += call_time

    ## Operations
    def __call__(self, bounds):
        raise NotImplementedError