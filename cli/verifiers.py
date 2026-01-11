import typing

import sys
sys.path.append('..')
import geometry.interval as interval
import verification.nn_verification as nn_verif
import verification.marabou as marabou_verif



def init_marabou_sound(
        c_star:             int,
        model_path_onnx:    str,
        domain:             interval.Interval,
        epsilon:            int =1
) -> nn_verif.NNVerification:
    
    return marabou_verif.SoundMarabouVerifier(c_star, model_path_onnx, domain, epsilon)



def init_marabou_complete(
        c_star:             int,
        model_path_onnx:    str,
        domain:             interval.Interval,
        epsilon:            int =1
) -> nn_verif.NNVerification:
    
    return marabou_verif.CompleteMarabouVerifier(c_star, model_path_onnx, domain, epsilon)



#################
# Verifiers Ids #
#################

# Marabou Verifiers
marabou_sound       = 0
marabou_complete    = 1

## Types, types, types.. types everywhere
InitMethod_t = typing.Callable[
                [
                    int,
                    str,
                    interval.Interval,
                    int
                ],
                nn_verif.NNVerification
            ]

init_method: typing.Dict[int, InitMethod_t] = {
    marabou_sound:      init_marabou_sound,
    marabou_complete:   init_marabou_complete
}