#############
# Libraries #
#############
# custom libraries
import sys
sys.path.append('..')
import montecarlo.utilities as mc

import verification.nn_verification as nn_verif



class ExperimentalVerification(nn_verif.NNVerification):
    def __init__(self, c_star, nn_model, num_samples=1000):
        super().__init__(c_star, nn_model)

        ## Experimental parameters
        self.num_samples = num_samples
    
    def __call__(self, bounds):
        counterexample = mc.adversarial_attack(
                            bounds.lb,
                            bounds.ub,
                            self.model_description,
                            self.c_star,
                            self.num_samples
                        )
        return (counterexample is None), counterexample