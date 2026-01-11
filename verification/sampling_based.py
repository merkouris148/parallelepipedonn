#############
# Libraries #
#############
# custom libraries
import sys
sys.path.append('..')
import montecarlo.utilities as mc

import verification.nn_verification as nn_verif



class SamplingBasedVerification(nn_verif.NNVerification):
    def __init__(self, c_star, samples):
        super().__init__(c_star, samples)
    
    def __call__(self, bounds):
        for c in self.model_description.keys():
            if c == self.c_star: continue

            point_set = self.model_description[c]
            for point in point_set:
                if point in bounds: return False, point
            
        return True, None