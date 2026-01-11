#############
# Libraries #
#############
# python libraries
import time

# 3rd party libraries
import numpy as np

from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils

# custom libraries
import sys
sys.path.append('..')
import verification.nn_verification as nn_verif

from geometry.constants import epsilon


#############
# Constants #
#############
sat     = 0
unsat   = 1
timeout = 2
marabou_retvals = [
                    "sat",
                    "unsat",
                    "TIMEOUT"
                ]


## Exception handling
# If the counterexample belong's to the same class
# as x_star, we cannot proceed with the explanation's
# refinement. Thus we treat this oracle call as 'sat'
# and terminate the search returning the current results

## treat wrong class as sat
wrong_class_sat = True
## exit codes
wrong_class_exit_code = 10

####################
# Helper Functions #
####################
def marabou2numpy(counterexample_dict, select_first_n, row_dim, column_dim):
    return np.array(list(counterexample_dict.values()))[0:select_first_n].reshape((row_dim, column_dim))




####################
# Marabou Verifier #
####################


## NN Varification oracle
# a wrapper for Marabou
class MarabouVerification(nn_verif.NNVerification):
    ## Constructor
    # NOTE: * Marabou seems to be *extremely* sensitive to the epsilon
    # value. Note that the epsilon defined here, differs from:
    # geometry.constants.epsilon.
    # * epsilon here is the slack between the maximum score of the
    # predictor and the rest of the values
    def __init__(self, c_star, model_path_onnx, domain, epsilon=1):
        ## Initialize super class
        super().__init__(c_star, Marabou.read_onnx(model_path_onnx))

        ## create options
        self.options = Marabou.createOptions(   
                                                numWorkers      = 8,
                                                timeoutInSeconds= 120,
                                                verbosity       = 0,
                                                # BE CAREFUL Gurobi does NOT support
                                                # disjunction of constraints.
                                                # DO NOT set to True!
                                                solveWithMILP=False
                                            )

        ## get the *symbolic* I/O variables from marabou
        self.inputVars   = self.model_description.inputVars[0][0]
        self.outputVars  = self.model_description.outputVars[0][0]

        ## Dimensions
        self.row_dim    = self.inputVars.shape[0]
        self.column_dim = self.inputVars.shape[1]
        self.dim        = self.row_dim * self.column_dim

        ## Domain
        self.domain = domain
        # Set input constraints
        for i in range(self.inputVars.shape[0]):
            for j in range(self.inputVars.shape[1]):
                self.model_description.setUpperBound(
                                                self.inputVars[i][j],
                                                domain.ub[i][j]
                                            )
                self.model_description.setLowerBound(
                                                self.inputVars[i][j],
                                                domain.lb[i][j]
                                            )

        ## Parameters
        self.epsilon = epsilon  # minimum slack between the 1st and
                                # the 2nd class score


    ## Predicates
    def check_witness(self, witness, bounds):
        raise NotImplementedError()

    ## Predictions
    # * The second argument in the evaluate() method is a flag for
    # using Marabou for the evaluation.
    # * Using Marabou to make prediction is *completely* unstable!
    # so its usage is discouraged.
    # * The last argument of the evaluate() method is a path to
    # redirect the output of the evaluation method.
    # * We pass an empty string, since we set verbose=False at the
    # options. Thus, keeping the evaluation silent.
    def predict(self, X):
        return self.model_description.evaluate([X], False, self.options, "")
    
    def predict_argmax(self, X):
        predictions_vector  = self.predict(X)[0][0]
        prediction_class    = np.argmax(predictions_vector)
        prediction_value    = predictions_vector[prediction_class]

        return prediction_class, prediction_value



    ## Operations
    ## Checking Explanation's Soundness
    def __call__(self, bounds):
        marabou_tic = time.time()
        marabou_val = self.model_description.solve(options=self.options, verbose=False)
        marabou_toc = time.time()
        self.set_statistics(marabou_toc - marabou_tic)
        assert marabou_val[0] in marabou_retvals, \
        ("Marabou Unexpected Ret. Val.:", marabou_val[0])
        

        ## Return Values
        # We handle timeout as unsat. This breaks soundness of our algorithm
        if  (marabou_val[0] == marabou_retvals[unsat]) or\
            (marabou_val[0] == marabou_retvals[timeout]):

            if marabou_val[0] == marabou_retvals[timeout]: self.num_timeouts += 1

            return True, None
        
        else:
            witness = marabou2numpy(marabou_val[1], self.dim, self.row_dim, self.column_dim)
            self.check_witness(witness, bounds)

            return False, witness




##########################
# Sound Marabou Verifier #
##########################
class SoundMarabouVerifier(MarabouVerification):
    def __init__(self, c_star, model_path_onnx, domain, epsilon=1):
        super().__init__(c_star, model_path_onnx, domain, epsilon)

        ## Set Ouput Constraints For Negative Counter Examples
        # i.e. Adversarial Examples
        # at least one of the output variables exeeding the correct class
        out_constraints = []
        for y in range(len(self.outputVars)):
            # for y != c_star
            if y == self.c_star: continue

            # create new output constraint: <=> y_i - y_{c_star} >= e
            new_out_constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            new_out_constraint.addAddend(1.0, self.outputVars[y])
            new_out_constraint.addAddend(-1.0, self.outputVars[self.c_star])
            new_out_constraint.setScalar(1.0 * self.epsilon)


            out_constraints.append([new_out_constraint])
        
        # \/_{i != c_star} [y_i - y_{c_star} >= e]
        self.model_description.addDisjunctionConstraint(out_constraints)

    ###############
    # Call Method #
    ###############
    def __call__(self, bounds):
        ## set input constraints
        for i in range(self.inputVars.shape[0]):
            for j in range(self.inputVars.shape[1]):
                self.model_description.setUpperBound(
                                                self.inputVars[i][j],
                                                bounds.ub[i][j]
                                            )
                self.model_description.setLowerBound(
                                                self.inputVars[i][j],
                                                bounds.lb[i][j]
                                            )

        return super().__call__(bounds)

    ###########################
    # Check Marabou's Witness #
    ###########################
    def check_witness(self, witness, bounds):
        prediction, _ =  self.predict_argmax(witness)

        ## make sure that the counter example is indeed a counterexample:
        # a) counterexample's class is different than c_star
        # b) counterexample belongs to the bounds
        assert witness in bounds, "Witness not in bounds"
        ## Some exception handling
        # If the counterexample belong's to the same class
        # as x_star, we cannot proceed with the explanation's
        # refinement. Thus we treat this oracle call as 'sat'
        # and terminate the search returning the current results
        if prediction == self.c_star:
            print(
                "Witness' class: ", str(prediction), "c_star:", self.c_star,
                file=sys.stderr
            )
            if wrong_class_sat:
                print(
                    "Treating wrong class as sat",
                    file=sys.stderr
                )
                return True, None
            else:
                exit(wrong_class_exit_code)



#############################
# Complete Marabou Verifier #
#############################
class CompleteMarabouVerifier(MarabouVerification):
    def __init__(self, c_star, model_path_onnx, domain, epsilon=1.0):
        super().__init__(c_star, model_path_onnx, domain, epsilon)

        for y in range(len(self.outputVars)):
            # for y != c_star
            if y == self.c_star: continue

            self.model_description.addInequality(
                [self.outputVars[y], self.outputVars[self.c_star]],
                [1.0, -1.0],
                -1.0 * self.epsilon
            )
    

    ###############
    # Call Method #
    ###############
    def __call__(self, bounds):
        # clear previous disjunctions
        self.model_description.disjunctionList = []
        out_constraints = []

        ## set input constraints
        for i in range(self.inputVars.shape[0]):
            for j in range(self.inputVars.shape[1]):
                
                # lower bound constraints
                lb_new_out_constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
                lb_new_out_constraint.addAddend(1.0, self.inputVars[i][j])
                lb_new_out_constraint.setScalar(1.0 * bounds.lb[i][j] - 1e-1)
                out_constraints.append([lb_new_out_constraint])

                # upper bound constraints
                ub_new_out_constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
                ub_new_out_constraint.addAddend(1.0, self.inputVars[i][j])
                ub_new_out_constraint.setScalar(1.0 * bounds.ub[i][j] + 1e-1)
                out_constraints.append([ub_new_out_constraint])
        
        
        self.model_description.addDisjunctionConstraint(out_constraints)

        return super().__call__(bounds)

    ###########################
    # Check Marabou's Witness #
    ###########################
    def check_witness(self, witness, bounds):
        prediction, _ =  self.predict_argmax(witness)

        ## make sure that the counter example is indeed a counterexample:
        # a) counterexample's class is different than c_star
        # b) counterexample belongs to the bounds
        assert not (witness in bounds), "Witness in bounds"
        ## Some exception handling
        # If the counterexample belong's to the same class
        # as x_star, we cannot proceed with the explanation's
        # refinement. Thus we treat this oracle call as 'sat'
        # and terminate the search returning the current results
        if prediction != self.c_star:
            print(
                "Witness' class: ", str(prediction), "c_star:", self.c_star,
                file=sys.stderr
            )
            if wrong_class_sat:
                print(
                    "Treating wrong class as sat",
                    file=sys.stderr
                )
                return True, None
            else:
                exit(wrong_class_exit_code)