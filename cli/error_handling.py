import sys

import cli.args as args

## error codes
error_all_ok                        = 0

# x_star
error_x_star_missing                = 2
error_x_star_no_file                = 3

# c_star
error_c_star_missing                = 4
error_c_star_no_int                 = 5

# onnx
error_onnx_path_missing             = 6
error_onnx_path_no_file             = 7

# lb, ub paths
error_lb_file_missing               = 8
error_ub_file_missing               = 9
error_bounds_from_file_non_parallel = 10

# output dir
error_out_dir_no_str                = 11

# algos
error_unknown_method                = 12
error_max_it_not_int                = 13
error_rad_no_float                  = 14
error_delta_no_float                = 15
error_delta_g_radius                = 20
error_dom_lb_no_float               = 16
error_dom_ub_no_float               = 17

# interface
error_unknown_help_arg              = 18

# unkwon error
error_unkown_error                  = 19

# verifiers
error_unknown_verif                 = 20

# timer
error_timer_not_pos_int             = 21



error_messages = {
    ## Required Args
    # x_star
    error_x_star_missing:                   "The required argument " + args.cli_args[args.required][args.x_star_path] + " was not given!",
    error_x_star_no_file:                   "The given x_star path does not exists!",

    # c_star
    error_c_star_missing:                   "The required argument " + args.cli_args[args.required][args.c_star] + " was not given!",
    error_c_star_no_int:                    "The given c_star is not an integer!",

    # onnx path
    error_onnx_path_missing:                "The required argument " + args.cli_args[args.required][args.onnx_path] + " was not given!",
    error_onnx_path_no_file:                "The given onnx description path does not exists!",

    ## Optional Args
    # lb, ub paths
    error_lb_file_missing:                  "The given lower bound path does not exists!",
    error_ub_file_missing:                  "The given upper bound path does not exists!",
    error_bounds_from_file_non_parallel:    "Bounds from file option given, but not parallelepipedal algorithm selected!",

    # output dir
    error_out_dir_no_str:                   "The given output dir name is not a string!",

    #verifiers
    error_unknown_verif:                     "Unknown given verifier!",

    # algos
    error_unknown_method:                   "Unknown given method!",
    error_max_it_not_int:                   "Max. number of iterations is not an integer!",
    error_rad_no_float:                     "Radius of distance restriction in not a float!",
    error_delta_no_float:                   "The percision constant delta in not a float!",
    error_delta_g_radius:                   "Delta greater than radius!",
    error_dom_lb_no_float:                  "The domain lower bound parameter in not a float!",
    error_dom_ub_no_float:                  "The domain upper bound parameter in not a float!",
    error_timer_not_pos_int:                "Timeout is not a positive integer!",

    # interface
    error_unknown_help_arg:                 "Unknown help argument!",

    # unkown error
    error_unkown_error:                     "Unknown error!"
}


default_suggestion = "\tUse 'python parallelepipedonn.py -h' for help.\n"

def print_error_message(error_code: int):
    assert error_code in error_messages.keys()

    print("\nError: " + error_messages[error_code], file=sys.stderr)
    print(default_suggestion, file=sys.stderr)
    exit(error_code)


warning_class_oracle_inconcistency  = 100
warning_overwrite_given_class       = 101

warning_messages ={
    warning_class_oracle_inconcistency: "Verif. oracle -- neural network inconcistency!",
    warning_overwrite_given_class:      "Ovewritting given class, replacing it with oracle's prediction."
}

def print_warning_message(warning_code: int, msg: str = ""):
    assert warning_code in warning_messages.keys()

    print("\nWarning: " + warning_messages[warning_code], file=sys.stderr)
    if msg != "": print(msg, file=sys.stderr)