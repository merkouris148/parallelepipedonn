## libraries
# python libraries
import math
import threading
import os
import subprocess
from datetime import datetime

# custom libraries
import sys
sys.path.append("..")
import cli.application as app
import cli.methods as methods
import cli.args as args
import cli.verifiers as verifs


## constants
predictions_standard_filename = "/predictions.txt"
explainer_command = "../bin/parallelepipedonn.py"

####################
# Helper Functions #
####################

file_sort_key = lambda filename:\
    int(filename.split("_")[0])\
    * 10**len(filename)\
    + int(filename.split("_")[1].split(".")[0])


#########
# Class #
#########


class ArgumentVector:
    def __init__(
            self,
            x_star_path,
            c_star,
            onnx_path,
            method_pfx,
            max_it,
            timeout,
            rad,
            delta,
            dom_lb,
            dom_ub,
            lower_bound,
            upper_bound
        ):
    
        ## Initialization
        # NN parameters
        self.x_star_path        = x_star_path
        self.c_star             = c_star
        self.onnx_path          = onnx_path
            
        # context
        self.method_pfx         = method_pfx
        self.max_it             = max_it
        self.timeout            = timeout
        self.rad                = rad
        self.delta              = delta
        self.dom_lb             = dom_lb
        self.dom_ub             = dom_ub
        self.lower_bound        = lower_bound
        self.upper_bound        = upper_bound
        
        # set appropriate verifier
        self.verif              = ""
        if self.method_pfx == args.algo_args[methods.complete_bu] or\
            self.method_pfx == args.algo_args[methods.complete_c_d_bu]:
            self.verif = args.verif_args[verifs.marabou_complete]
        else:
            self.verif = args.verif_args[verifs.marabou_sound]

    def get_argv(self):
        ## Temporary FIX, should correct later
        command_str =  "python"                         + " "
        command_str +=  explainer_command               + " "
        command_str +=  args.cli_args[args.required][args.x_star_path] + " " + self.x_star_path    + " "
        command_str +=  args.cli_args[args.required][args.c_star]      + " " + str(self.c_star)    + " "
        command_str +=  args.cli_args[args.required][args.onnx_path]   + " " + self.onnx_path      + " "
        command_str +=  args.cli_args[args.optional][args.method]      + " " + self.method_pfx     + " "
        command_str +=  args.cli_args[args.optional][args.max_it]      + " " + str(self.max_it)    + " "
        command_str +=  args.cli_args[args.optional][args.timeout]     + " " + str(self.timeout)   + " "
        command_str +=  args.cli_args[args.optional][args.rad]         + " " + str(self.rad)       + " "
        command_str +=  args.cli_args[args.optional][args.delta]       + " " + str(self.delta)     + " "
        command_str +=  args.cli_args[args.optional][args.dom_lb]      + " " + str(self.dom_lb)    + " "
        command_str +=  args.cli_args[args.optional][args.dom_ub]      + " " + str(self.dom_ub)    + " "
        command_str +=  args.cli_args[args.optional][args.verif]       + " " + self.verif          + " "

        # if appropriate, set -lb, -ub arguments
        if self.lower_bound != "" and self.upper_bound != "":
            command_str +=  args.cli_args[args.optional][args.lb_path]     + " " + self.lower_bound    + " "
            command_str +=  args.cli_args[args.optional][args.ub_path]     + " " + self.upper_bound    + " "
        
        ## suffix
        command_str += args.cli_args[args.optional][args.quiet]        + " "
        command_str += args.cli_args[args.optional][args.simple_res]   + " "
        command_str += args.cli_args[args.optional][args.save_images]  + " "
        command_str += args.cli_args[args.optional][args.over_out]     + " "

        return command_str

    def get_argv_list(self):
        ## Temporary FIX, should correct later
        command_list =  ["python"]
        command_list += [explainer_command]
        command_list += [args.cli_args[args.required][args.x_star_path],   self.x_star_path    ]
        command_list += [args.cli_args[args.required][args.c_star],        str(self.c_star)    ]
        command_list += [args.cli_args[args.required][args.onnx_path],     self.onnx_path      ]
        command_list += [args.cli_args[args.optional][args.method],        self.method_pfx     ]
        command_list += [args.cli_args[args.optional][args.max_it],        str(self.max_it)    ]
        command_list += [args.cli_args[args.optional][args.timeout],       str(self.timeout)   ]
        command_list += [args.cli_args[args.optional][args.rad],           str(self.rad)       ]
        command_list += [args.cli_args[args.optional][args.delta],         str(self.delta)     ]
        command_list += [args.cli_args[args.optional][args.dom_lb],        str(self.dom_lb)    ]
        command_list += [args.cli_args[args.optional][args.dom_ub],        str(self.dom_ub)    ]
        command_list += [args.cli_args[args.optional][args.verif],         self.verif          ]

        # if appropriate, set -lb, -ub arguments
        if self.lower_bound != "" and self.upper_bound != "":
            command_list +=  [args.cli_args[args.optional][args.lb_path],  self.lower_bound]
            command_list +=  [args.cli_args[args.optional][args.ub_path],  self.upper_bound]
        
        ## suffix
        command_list += [args.cli_args[args.optional][args.quiet]       ]
        command_list += [args.cli_args[args.optional][args.simple_res]  ]
        command_list += [args.cli_args[args.optional][args.save_images] ]
        command_list += [args.cli_args[args.optional][args.over_out]    ]

        return command_list



class ResultsVector:
    def __init__(self, res_vec_str):
        #print("\nres vec", res_vec_str, "\n")
        lines   = res_vec_str.split("\n")
        tokens  = lines[0].split(" ")
        
        self.num_it             = int(tokens[0])
        self.time               = float(tokens[1])
        self.comp               = int(tokens[2])
        self.min_edge_len       = float(tokens[3])
        self.verif_tot_time     = float(tokens[4])
        self.verif_num_calls    = int(tokens[5])
        self.timeout            = int(tokens[6])



class Experiments:
    
    ## Initialization
    def __init__(
        self,
        
        ## NN parameters
        input_directory,    # path to the input dataset
        onnx_path,          # path to the nn

        ## Experimental parameters
        num_threads,         # number of threads

        ## Context
        method_pfx  = methods.top_down, # specified algorithm for explanations
        max_it      = 10000,            # max number of iterations
        timeout     = 60,
        bounds_dir  = "",
        rad         = 1,                # radius of distance restriction
        delta       = 0.1,              # percision constant
        dom_lb      = 0.0,              # instance domain
        dom_ub      = 1.0,
        res_log     = True,
        err_log     = True
    ):

        ## Preconditions
        # NN parameters
        assert onnx_path != ""
        assert input_directory != ""

        # Experimental parameters
        assert num_threads > 0

        # Context
        assert method_pfx in args.args_algo.keys()
        assert max_it > 0
        assert timeout > 0
        assert delta > 0
        assert rad > delta

        ## Initialization
        # NN parameters
        self.input_directory    = input_directory
        self.onnx_path          = onnx_path
        
        # Experimental parameters
        self.num_threads        = num_threads

        # context
        self.method_pfx         = method_pfx
        self.max_it             = max_it
        self.timeout            = timeout
        self.rad                = rad
        self.delta              = delta
        self.dom_lb             = dom_lb
        self.dom_ub             = dom_ub
        self.bounds_dir         = bounds_dir

        # get the file with the predictions
        self.predictions_path   = input_directory + predictions_standard_filename
        predict_file            = open(self.predictions_path, "r")
        self.predictions        = []
        for line in predict_file: self.predictions.append(int(line))
        
        ## Files
        # output dir
        self.output_dir = ""
        self.make_output_dir_path()
        # results log
        self.res_log        = res_log
        self.res_log_lock   = threading.Lock()
        self.res_log_path   = self.output_dir + "/" + "results.log"
        # subprocess stderr log
        self.err_log        = err_log
        self.err_log_lock   = threading.Lock()
        self.err_log_path   = self.output_dir + "/" + "errors.log"


        ## Experiments
        # The data-structures of the undone & done experiments.
        # Each time we perform an experiment, we remove it from the
        # undone and insert it to done.

        # the experiments not yet performed
        self.experiments         = []
        self.experiments_lock    = threading.Lock()
        i = 0
        in_files = list(os.listdir(self.input_directory))
        in_files = list(filter(lambda filename: filename.split(".")[-1] == "csv", in_files))
        in_files.sort(key=file_sort_key)
        
        for x_star_path_name in in_files:
            x_star_path_name_pfx = x_star_path_name.split(".")[0]
            x_star_path_name_sfx = x_star_path_name.split(".")[1]
            if x_star_path_name_sfx != "csv": continue
            
            lowerbound_path = ""
            upperbound_path = ""
            if self.bounds_dir != "":            
                lowerbound_path = self.bounds_dir + "/" + x_star_path_name_pfx + "_lb.csv"
                upperbound_path = self.bounds_dir + "/" + x_star_path_name_pfx + "_ub.csv"
                print(lowerbound_path)
                print(self.bounds_dir)

            arg_vec = ArgumentVector(
                            self.input_directory + "/" + x_star_path_name,
                            self.predictions[i],
                            self.onnx_path,
                            self.method_pfx,
                            self.max_it,
                            self.timeout,
                            self.rad,
                            self.delta,
                            self.dom_lb,
                            self.dom_ub,
                            lowerbound_path,
                            upperbound_path
                        )
            self.experiments.append(arg_vec)
            i += 1
        # the experiments that were performed
        self.results_lock  = threading.Lock()
        self.results       = []



        ## Constructing the Workers
        self.workers = []
        for i in range(self.num_threads):
            self.workers.append(threading.Thread(target=self._do_experiment))
        
        ## subprocess exit codes
        self.exit_codes_lock    = threading.Lock()
        self.exit_codes         = []


        ## States
        self.experiments_performed = False


        ## Statistics
        # results string
        self.str_res = None

        # num of iterations
        self.min_num_it         = math.inf
        self.avg_num_it         = 0
        self.max_num_it         = -math.inf
        self.var_num_it         = 0

        # time
        self.min_time       = math.inf
        self.avg_time       = 0
        self.max_time       = -math.inf
        self.var_time       = 0
        self.tot_time       = 0
        self.timeouts       = 0

        # guarantee's complexity
        self.min_comp       = math.inf
        self.avg_comp       = 0
        self.max_comp       = -math.inf
        self.var_comp       = 0

        # guarantees min edge length
        self.min_min_edge_len    = math.inf
        self.avg_min_edge_len    = 0
        self.max_min_edge_len    = -math.inf
        self.var_min_edge_len    = 0

        # verif time
        self.min_verif_time       = math.inf
        self.avg_verif_time       = 0
        self.max_verif_time       = -math.inf
        self.var_verif_time       = 0
        self.tot_verif_time       = 0

        # verif num calls
        self.min_verif_calls      = math.inf
        self.avg_verif_calls      = 0
        self.max_verif_calls      = -math.inf
        self.var_verif_calls      = 0
        self.tot_verif_calls      = 0

        # verif
        self.verif_time_call      = 0
        self.verif_percent        = 0

    
    ## Predicates
    def check_exit_codes(self):
        return self.exit_codes == []


    ## Operations
    # do the experiments
    def do_experiments(self):
        ## Precondition
        assert self.experiments_performed == False

        ## Start Workers
        for worker in self.workers: worker.start()

        ## Wait for them to finish
        for worker in self.workers: worker.join()

        ## Postcondition
        self.experiments_performed = True


    # perform a single experiment
    def _do_experiment(self):
        while True:
            # DEBUG
            #print("I am", threading.get_ident())

            # acquire lock
            self.experiments_lock.acquire()

            ## Check if there are experiments
            if self.experiments == []:
                self.experiments_lock.release()
                break

            ## If there are experiments to be made, do one
            arg_vec = self.experiments.pop()

            # some I/O
            print(
                "Thread:",      threading.get_ident(),
                "queue len.:",  len(self.experiments),
                "x_star path:", arg_vec.x_star_path
            )

            # release lock
            self.experiments_lock.release()

            ## spawn subprocess
            parallelepipedonn_call = subprocess.run(
                arg_vec.get_argv_list(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if self.err_log:
                self.writeln_err_log(
                    "\n" +\
                    arg_vec.x_star_path + ":\n" +\
                    parallelepipedonn_call.stderr +\
                    "\n"
                )

            if parallelepipedonn_call.returncode != 0:
                print("Thread:", threading.get_ident(), "Subprocess error!:", parallelepipedonn_call.returncode)
                print("Subprocess message:", parallelepipedonn_call.stderr)
                self.exit_codes_lock.acquire()
                self.exit_codes.append((
                        arg_vec.x_star_path,
                        parallelepipedonn_call.returncode
                    ))
                self.exit_codes_lock.release()
                break

            parallelepipedonn_call_out = parallelepipedonn_call.stdout
            #print("\n", threading.get_ident(), "subprocess:", parallelepipedonn_call_out, "\n")
            res_vec = ResultsVector(parallelepipedonn_call_out)
            

            ## Write experiments results
            self.results_lock.acquire()
            
            # some I/O
            if self.res_log: self.writeln_res_log(arg_vec.x_star_path + " " + parallelepipedonn_call_out)
 
            self.results.append(res_vec)
            self.results_lock.release()
    

    ## Statistics
    def calculate_statistics(self):
        #assert self.check_exit_codes()

        for res in self.results:
            # num iterations
            self.avg_num_it         += res.num_it
            self.max_num_it         = max(res.num_it, self.max_num_it)
            self.min_num_it         = min(res.num_it, self.min_num_it)

            # time
            self.tot_time   += res.time
            self.max_time   = max(res.time, self.max_time)
            self.min_time   = min(res.time, self.min_time)
            self.timeouts   += res.timeout

            # explanation's complexity
            self.avg_comp   += res.comp
            self.max_comp   = max(res.comp, self.max_comp)
            self.min_comp   = min(res.comp, self.min_comp)

            # explanation's max inf radius
            self.avg_min_edge_len   += res.min_edge_len
            self.max_min_edge_len   = max(res.min_edge_len, self.max_min_edge_len)
            self.min_min_edge_len   = min(res.min_edge_len, self.min_min_edge_len)

            # verif time
            self.tot_verif_time       += res.verif_tot_time
            self.max_verif_time       = max(res.verif_tot_time, self.max_verif_time)
            self.min_verif_time       = min(res.verif_tot_time, self.min_verif_time)

            # verif num calls
            self.tot_verif_calls      += res.verif_num_calls
            self.max_verif_calls      = max(res.verif_num_calls, self.max_verif_calls)
            self.min_verif_calls      = min(res.verif_num_calls, self.min_verif_calls)
            
        
        ## Scale avgs
        self.avg_num_it         /= len(self.results)
        self.avg_time           = self.tot_time / len(self.results)
        self.avg_comp           /= len(self.results)
        self.avg_min_edge_len   /= len(self.results)
        self.avg_verif_time     = self.tot_verif_time / len(self.results)
        self.avg_verif_calls    = self.tot_verif_calls / len(self.results)

        ## Calculate variance
        for res in self.results:
            self.var_num_it         += (self.avg_num_it - res.num_it)**2
            self.var_time           += (self.avg_time - res.time)**2
            self.var_comp           += (self.avg_comp - res.comp)**2
            self.var_min_edge_len   += (self.avg_min_edge_len - res.min_edge_len)**2
            self.var_verif_time     += (self.avg_verif_time - res.verif_tot_time)**2
            self.var_verif_calls    += (self.avg_verif_calls - res.verif_num_calls)**2
        
        ## Scale variances
        self.var_num_it         /= len(self.results)
        self.var_time           /= len(self.results)
        self.var_comp           /= len(self.results)
        self.var_min_edge_len   /= len(self.results)
        self.var_verif_time     /= len(self.results)
        self.var_verif_calls    /= len(self.results)

        ## Overal Verif Statistics
        self.verif_time_call = self.tot_verif_time / self.tot_verif_calls
        self.verif_percent   = self.tot_verif_time / self.tot_time

    
    # I/O
    def make_str_reults(self):
        self.str_res ="### Experimental Reults ###"                                     + "\n" +\
                    f"{'Date-Time:':<26}"           + str(datetime.now())               + "\n" +\
                    "# Parameters:"                                                     + "\n" +\
                    f"{'Method:':<26}"              + str(self.method_pfx)              + "\n" +\
                    f"{'Max. It.:':<26}"            + str(self.max_it)                  + "\n" +\
                    f"{'Radius Dist. Restr.:':<26}" + str(self.rad)                     + "\n" +\
                    f"{'Delta:':<26}"               + str(self.delta)                   + "\n" +\
                    "-" * 60                                                            + "\n" +\
                    "# Iterations:"                                                     + "\n" +\
                    f"{'Min.:':<26}"                + str(self.min_num_it)              + "\n" +\
                    f"{'Avg.:':<26}"                + str(self.avg_num_it)              + "\n" +\
                    f"{'Max.:':<26}"                + str(self.max_num_it)              + "\n" +\
                    f"{'Var.:':<26}"                + str(self.var_num_it)              + "\n" +\
                    f"{'Std Dev.:':<26}"            + str(math.sqrt(self.var_num_it))   + "\n" +\
                    "-" * 60                                                            + "\n" +\
                    "# Time:"                                                           + "\n" +\
                    f"{'Min.:':<26}"                + str(self.min_time)                + "\n" +\
                    f"{'Avg.:':<26}"                + str(self.avg_time)                + "\n" +\
                    f"{'Max.:':<26}"                + str(self.max_time)                + "\n" +\
                    f"{'Var.:':<26}"                + str(self.var_time)                + "\n" +\
                    f"{'Std Dev.:':<26}"            + str(math.sqrt(self.var_time))     + "\n" +\
                    f"{'Timeouts:':<26}"            + str(self.timeouts)                + "\n" +\
                    "-" * 60                                                            + "\n" +\
                    "# Guarantee's Description Complexity:"                             + "\n" +\
                    f"{'Min.:':<26}"                + str(self.min_comp)                + "\n" +\
                    f"{'Avg.:':<26}"                + str(self.avg_comp)                + "\n" +\
                    f"{'Max.:':<26}"                + str(self.max_comp)                + "\n" +\
                    f"{'Var.:':<26}"                + str(self.var_comp)                + "\n" +\
                    f"{'Std Dev.:':<26}"            + str(math.sqrt(self.var_comp))     + "\n" +\
                    "-" * 60                                                            + "\n" +\
                    "# Guarantee's Minimum Edge Length:"                                + "\n" +\
                    f"{'Min.:':<26}"                + str(self.min_min_edge_len)        + "\n" +\
                    f"{'Avg.:':<26}"                + str(self.avg_min_edge_len)        + "\n" +\
                    f"{'Max.:':<26}"                + str(self.max_min_edge_len)        + "\n" +\
                    f"{'Var.:':<26}"                + str(self.var_min_edge_len)        + "\n" +\
                    f"{'Std Dev.:':<26}"            + str(math.sqrt(self.var_min_edge_len)) + "\n" +\
                    "-" * 60                                                            + "\n" +\
                    "# Verification Oracle's Time:"                                     + "\n" +\
                    f"{'Min.:':<26}"                + str(self.min_verif_time)          + "\n" +\
                    f"{'Avg.:':<26}"                + str(self.avg_verif_time)          + "\n" +\
                    f"{'Max.:':<26}"                + str(self.max_verif_time)          + "\n" +\
                    f"{'Var.:':<26}"                + str(self.var_verif_time)          + "\n" +\
                    f"{'Std Dev.:':<26}"            + str(math.sqrt(self.var_verif_time)) + "\n" +\
                    "-" * 60                                                            + "\n" +\
                    "# Verification Oracle Num. of Calls:"                              + "\n" +\
                    f"{'Min.:':<26}"                + str(self.min_verif_calls)         + "\n" +\
                    f"{'Avg.:':<26}"                + str(self.avg_verif_calls)         + "\n" +\
                    f"{'Max.:':<26}"                + str(self.max_verif_calls)         + "\n" +\
                    f"{'Var.:':<26}"                + str(self.var_verif_calls)         + "\n" +\
                    f"{'Std Dev.:':<26}"            + str(math.sqrt(self.var_verif_calls)) + "\n" +\
                    "-" * 60                                                            + "\n" +\
                    "# Verification Overall Statistics"                                 + "\n" +\
                    f"{'Avg. Verif. Time/Call:':<26}"      + str(self.verif_time_call)  + "\n" +\
                    f"{'Percent. Verif_t/Tot_t:':<26}"     + str(self.verif_percent)



    def print_results(self):
        print(self.str_res)
    
    def make_output_dir_path(self):
        tokens      = self.input_directory.split("/")
        path_header = "/".join(tokens[:-1])
        
        self.output_dir  =  path_header         + "/" +\
                            "outputs"           + "/" +\
                            self.method_pfx
        
        ## hope this works
        os.makedirs(self.output_dir, exist_ok=True)

    
    def writeln_res_log(self, res_str):
        self.res_log_lock.acquire()

        f = open(self.res_log_path, "a")
        f.write(res_str)
        f.close()

        self.res_log_lock.release()
    

    def writeln_err_log(self, err_str):
        self.res_log_lock.acquire()

        f = open(self.err_log_path, "a")
        f.write(err_str)
        f.close()

        self.res_log_lock.release()


    def save_results(self):
        self.make_str_reults()

        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)

        output_path = self.output_dir + "/" + "results.txt"
        f = open(output_path, "w")
        f.write(self.str_res)
        f.close()
