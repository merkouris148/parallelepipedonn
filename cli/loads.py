## Typing
import typing

## Paths
import os

## Arguments
import cli.args as args
import cli.methods as methods


###################
# Check Arguments #
###################

## Generic Required Arguments
def load_required_str(argv: typing.List[str], arg: str) -> str:
    if args.cli_args[args.optional][args.help] in argv: return None

    return argv[argv.index(arg) + 1]

def load_required_int(argv: typing.List[str], arg: str) -> int:
    if args.cli_args[args.optional][args.help] in argv: return None

    return int(argv[argv.index(arg) + 1])



## Generic Optional Arguments
def load_optional_str(argv: typing.List[str], arg: str) -> typing.Union[str, None]:
    if arg in argv: return argv[argv.index(arg) + 1]
    return None

def load_optional_int(argv: typing.List[str], arg: str) -> typing.Union[int, None]:
    if arg in argv: return int(argv[argv.index(arg) + 1])
    return None

def load_optional_float(argv: typing.List[str], arg: str) -> typing.Union[float, None]:
    if arg in argv: return float(argv[argv.index(arg) + 1])
    return None

def load_optional_bool(argv: typing.List[str], arg: str) -> typing.Union[bool, None]:
    if arg in argv: return True
    return None


## load radius restriction
def load_radius(argv: typing.List[str]) -> typing.Union[bool, None]:
    if  args.cli_args[args.optional][args.dom_lb] in argv and\
        args.cli_args[args.optional][args.dom_ub] in argv and\
        not args.cli_args[args.optional][args.rad] in argv:

        dom_lb = float(argv[argv.index(args.cli_args[args.optional][args.dom_lb]) + 1])
        dom_ub = float(argv[argv.index(args.cli_args[args.optional][args.dom_ub]) + 1])

        return dom_ub - dom_lb

    return None


## Verification
def load_verif(argv: typing.List[str]) -> typing.Union[int, None]:
    if args.cli_args[args.optional][args.verif] in argv:
        return args.args_verif[argv[argv.index(args.cli_args[args.optional][args.verif]) + 1]]
    return None


## Algorithm
def load_method(argv: typing.List[str]) -> typing.Union[int, None]:
    if args.cli_args[args.optional][args.method] in argv:
        return args.args_algo[argv[argv.index(args.cli_args[args.optional][args.method]) + 1]]
    return None


## Max. Iterations
def load_max_it(argv: typing.List[str]) -> typing.Union[int, None]:
    dom_lb = args.defaults[args.optional][args.dom_lb]
    dom_ub = args.defaults[args.optional][args.dom_ub]

    if  args.cli_args[args.optional][args.dom_lb] in argv and\
        args.cli_args[args.optional][args.dom_ub] in argv and\
        not args.cli_args[args.optional][args.rad] in argv:

        dom_lb = float(argv[argv.index(args.cli_args[args.optional][args.dom_lb]) + 1])
        dom_ub = float(argv[argv.index(args.cli_args[args.optional][args.dom_ub]) + 1])

    radius = dom_ub - dom_lb

    delta = args.defaults[args.optional][args.delta]
    if args.cli_args[args.optional][args.delta] in argv:
        delta = float(argv[argv.index(args.cli_args[args.optional][args.delta]) + 1])

    if  not args.cli_args[args.optional][args.max_it] in argv           and\
        args.cli_args[args.optional][args.method] in argv               and\
        argv[argv.index(args.cli_args[args.optional][args.method]) + 1] !=\
        args.algo_args[methods.top_down]                                and\
        argv[argv.index(args.cli_args[args.optional][args.method]) + 1] !=\
        args.algo_args[methods.complete_bu]:

        return int(radius / delta)
    
    elif args.cli_args[args.optional][args.max_it] in argv:
        return int(argv[argv.index(args.cli_args[args.optional][args.max_it]) + 1])
    

    return None




## Output dir
def load_out_dir(argv: typing.List[str]) -> typing.Union[str, None]:
    """
        #### Description
        Let
            `<path_header>/<dataset_dir>/inputs/<basename>.csv`
        be the path of the `x*` input.
        This function returns the prefix of the output's path, i.e.:
            `<path_header>/<dataset_dir>/outputs/<outputs_subdir>/<basename>`
        <br />Hence, the output will be:
            `<path_header>/<dataset_dir>/outputs/<outputs_subdir>/<basename>_<lb | ub>.csv`
        
        #### Special Behavior
        If the `-o <given_subdir>` argument is given, then,
            `<outputs_subdir> := <given_subdir>`
        else,
            `<outputs_subdir> := <algo_arg>`
        <br />
        If `<path_header>/<dataset_dir>/outputs/<outputs_subdir>/`<br />
        does not exists, it will be created.

        #### Notes
        If the `-si` (save images) bound is given, then the output path,
        will be:
            `<path_header>/<dataset_dir>/outputs/<outputs_subdir>/<basename>_<lb | ub>.png`
    """
    if args.cli_args[args.optional][args.help] in argv: return None

    # get the input path
    path           = argv[argv.index(args.cli_args[args.required][args.x_star_path]) + 1]
    
    # tokenize the input path
    basename       = path.split(".")[-2].split("/")[-1]      # the filename
    path_header    = "/".join(path.split("/")[:-2])         # the path header
                                                            # until before the input directory
    
    # handle output subdir
    # default behavior (no algorithm was given)
    output_subdir = args.algo_args[methods.top_down]
    
    # default behavior, when algorithm is given 
    if args.cli_args[args.optional][args.method] in argv:
        output_subdir = argv[argv.index(args.cli_args[args.optional][args.method]) + 1]
    
    # behavior when output subdir option is given
    if args.cli_args[args.optional][args.out_dir] in argv:
        # if given, replace the default output_subdir
        # with the given subdir
        output_subdir = argv[argv.index(args.cli_args[args.optional][args.out_dir]) + 1]

    ## Construct the output path
    output_dir = "/".join([path_header, "outputs", output_subdir])
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = "/".join([path_header, "outputs", output_subdir, basename])

    return output_prefix



## Help
def load_help(argv: typing.List[str]) -> typing.Union[int, None]:
    if not args.cli_args[args.optional][args.help] in argv:   return None                         # no help argument

    ind = argv.index(args.cli_args[args.optional][args.help]) + 1
    if ind == len(argv):        return args.help_default            # no help sub argument
    if args.is_arg(argv, ind):  return args.help_default            # no help sub argument
    else:                       return args.args_help[argv[ind]]    # load help sub argument



load = {
    args.required: {
        args.x_star_path: lambda argv: load_required_str(argv, args.cli_args[args.required][args.x_star_path]),
        args.c_star:      lambda argv: load_required_int(argv, args.cli_args[args.required][args.c_star]),
        args.onnx_path:   lambda argv: load_required_str(argv, args.cli_args[args.required][args.onnx_path])
    },
    
    args.optional: {
        # Input
        args.lb_path:     lambda argv: load_optional_str(argv, args.cli_args[args.optional][args.lb_path]),
        args.ub_path:     lambda argv: load_optional_str(argv, args.cli_args[args.optional][args.ub_path]),
        
        # Output
        args.out_dir:     load_out_dir,
        args.over_out:    lambda argv: load_optional_bool(argv, args.cli_args[args.optional][args.over_out]),
        args.save_images: lambda argv: load_optional_bool(argv, args.cli_args[args.optional][args.save_images]),

        # Algorithm
        args.verif:       load_verif,
        args.method:      load_method,
        args.max_it:      load_max_it,
        args.rad:         load_radius,
        args.delta:       lambda argv: load_optional_float(argv, args.cli_args[args.optional][args.delta]),
        args.dom_lb:      lambda argv: load_optional_float(argv, args.cli_args[args.optional][args.dom_lb]),
        args.dom_ub:      lambda argv: load_optional_float(argv, args.cli_args[args.optional][args.dom_ub]),
        args.timeout:     lambda argv: load_optional_int(argv, args.cli_args[args.optional][args.timeout]),
        
        # Interface
        args.no_out:      lambda argv: load_optional_bool(argv, args.cli_args[args.optional][args.no_out]),
        args.simple_res:  lambda argv: load_optional_bool(argv, args.cli_args[args.optional][args.simple_res]),
        args.quiet:       lambda argv: load_optional_bool(argv, args.cli_args[args.optional][args.quiet]),
        args.help:        load_help,
    }
}