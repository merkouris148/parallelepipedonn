## Typing
import typing

## Paths
import os

## Arguments
import cli.args as args
## Errors
import cli.error_handling as errors
## Utils
import cli.utils as utils

###################
# Check Arguments #
###################

## No errors
def check_no_errors(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    return False, errors.error_all_ok


## Required Arguments
def check_x_star_path(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                              return False, errors.error_all_ok

    if not args.cli_args[args.required][args.x_star_path] in argv:                                   return True, errors.error_x_star_missing
    if not (os.path.isfile(argv[argv.index(args.cli_args[args.required][args.x_star_path]) + 1])):   return True, errors.error_x_star_no_file

    return False, errors.error_all_ok


def check_c_star(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                             return False, errors.error_all_ok

    if not args.cli_args[args.required][args.c_star] in argv:                                       return True, errors.error_c_star_missing
    if not argv[argv.index(args.cli_args[args.required][args.c_star]) + 1].isnumeric():   return True, errors.error_c_star_no_int

    return False, errors.error_all_ok


def check_onnx_path(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                            return False, errors.error_all_ok

    if not args.cli_args[args.required][args.onnx_path] in argv:                                   return True, errors.error_onnx_path_missing
    if not (os.path.isfile(argv[argv.index(args.cli_args[args.required][args.onnx_path]) + 1])):   return True, errors.error_onnx_path_no_file

    return False, errors.error_all_ok


## Optional Arguments
# Input
def check_lb_path(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                          return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.lb_path] in argv:                                   return False, errors.error_all_ok
    if not (os.path.isfile(argv[argv.index(args.cli_args[args.optional][args.lb_path]) + 1])):   return True,  errors.error_lb_file_missing

    return False, errors.error_all_ok


def check_ub_path(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                          return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.ub_path] in argv:                                   return False, errors.error_all_ok
    if not (os.path.isfile(argv[argv.index(args.cli_args[args.optional][args.ub_path]) + 1])):   return True,  errors.error_ub_file_missing

    return False, errors.error_all_ok


# Output
def check_out_dir(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                 return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.out_dir] in argv:                          return False, errors.error_all_ok
    if argv[argv.index(args.cli_args[args.optional][args.out_dir]) + 1].isnumeric():    return True,  errors.error_out_dir_no_str

    return False, errors.error_all_ok


# def check_over_out(argv: typing.List[str]) -> typing.Tuple[bool, int]:
#     return check_no_errors(argv)

# def check_save_images(argv: typing.List[str]) -> typing.Tuple[bool, int]:
#     return check_no_errors(argv)


# Verifiers
def check_verifier(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                                return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.verif] in argv:                                           return False, errors.error_all_ok
    if not (argv[argv.index(args.cli_args[args.optional][args.verif]) + 1] in args.args_verif.keys()): return True,  errors.error_unknown_verif

    return False, errors.error_all_ok


# Algorithm
def check_method(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                                return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.method] in argv:                                          return False, errors.error_all_ok
    if not (argv[argv.index(args.cli_args[args.optional][args.method]) + 1] in args.args_algo.keys()): return True,  errors.error_unknown_method

    return False, errors.error_all_ok


def check_max_it(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                             return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.max_it] in argv:                                       return False, errors.error_all_ok
    if not argv[argv.index(args.cli_args[args.optional][args.max_it]) + 1].isnumeric():   return True,  errors.error_max_it_not_int

    return False, errors.error_all_ok


def check_rad(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                 return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.rad] in argv:                              return False, errors.error_all_ok
    if not utils.isfloat(argv[argv.index(args.cli_args[args.optional][args.rad]) + 1]): return True,  errors.error_rad_no_float

    return False, errors.error_all_ok


def check_delta(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                     return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.delta] in argv:                                return False, errors.error_all_ok
    if not utils.isfloat(argv[argv.index(args.cli_args[args.optional][args.delta]) + 1]):   return True,  errors.error_delta_no_float

    delta   = float(argv[argv.index(args.cli_args[args.optional][args.delta]) + 1])
    radius  = args.defaults[args.optional][args.rad]
    if args.cli_args[args.optional][args.rad] in argv:
        radius = float(argv[argv.index(args.cli_args[args.optional][args.rad]) + 1])
    if delta > radius:
        return True, errors.error_delta_g_radius

    return False, errors.error_all_ok


def check_dom_lb(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                     return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.dom_lb] in argv:                               return False, errors.error_all_ok
    if not utils.isfloat(argv[argv.index(args.cli_args[args.optional][args.dom_lb]) + 1]):  return True,  errors.error_dom_lb_no_float

    return False, errors.error_all_ok


def check_dom_ub(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    # overwrite checks if help arg is provided
    if args.cli_args[args.optional][args.help] in argv:                                     return False, errors.error_all_ok

    if not args.cli_args[args.optional][args.dom_ub] in argv:                               return False, errors.error_all_ok
    if not utils.isfloat(argv[argv.index(args.cli_args[args.optional][args.dom_ub]) + 1]):  return True,  errors.error_dom_ub_no_float

    return False, errors.error_all_ok


# Interface
# def check_no_out(argv: typing.List[str]) -> typing.Tuple[bool, int]:
#     return check_no_errors(argv)

# def check_simple_ress(argv: typing.List[str]) -> typing.Tuple[bool, int]:
#     return check_no_errors(argv)

# def check_quiet(argv: typing.List[str]) -> typing.Tuple[bool, int]:
#     return check_no_errors(argv)

def check_help(argv: typing.List[str]) -> typing.Tuple[bool, int]:
    if not args.cli_args[args.optional][args.help] in argv:    return False, errors.error_all_ok   # no -h argument was given

    ind = argv.index(args.cli_args[args.optional][args.help]) + 1
    if ind >= len(argv):                        return False, errors.error_all_ok   # no -h <sub arg> was given
    if args.is_arg(argv, ind):                  return False, errors.error_all_ok   # no -h <sub arg> was given
    if argv[ind] in args.args_help.keys():      return False, errors.error_all_ok   # known argument given
    
    # unkwon arguments given
    return True, errors.error_unknown_help_arg
    

check_argument = {
    ## Required Args
    args.required: {
        args.x_star_path:    check_x_star_path,
        args.c_star:         check_c_star,
        args.onnx_path:      check_onnx_path
    },

    ## Optional Args
    args.optional: {
        # Input
        args.lb_path:        check_lb_path,
        args.ub_path:        check_ub_path,
        
        # Output
        args.out_dir:        check_out_dir,
        args.over_out:       check_no_errors,
        args.save_images:    check_no_errors,

        # Verifiers
        args.verif:          check_verifier,

        # Algorithm
        args.method:         check_method,
        args.max_it:         check_max_it,
        args.rad:            check_rad,
        args.delta:          check_delta,
        args.dom_ub:         check_dom_lb,
        args.dom_lb:         check_dom_ub,
        
        # Interface
        args.no_out:         check_no_errors,
        args.simple_res:     check_no_errors,
        args.quiet:          check_no_errors,
        args.help:           check_help,
    }
}