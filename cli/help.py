import cli.methods as methods
import cli.verifiers as verifiers
import cli.args as args

help_required_header = "# Required Arguments:"
help_optional_header = "# Optional Arguments:"
help_msg = {
    # Required
    args.required: {
        args.x_star_path:   "the path to input point x_star",
        args.c_star:        "the class c_star of the input point x_star",
        args.onnx_path:     "the path to the onnx representation of a NN"
    },

    # Optional
    args.optional: {
        # Input
        args.lb_path:       "the path to lower bound .csv",
        args.ub_path:       "the path to upper bound .csv",

        # Output
        args.out_dir:       "output sub-directory, under /outoputs/<subdir>/out.csv",
        args.over_out:      "overwrite result-files if exist",
        args.save_images:   "save ub, lb as .png images, using output_dir",

        # verifiers
        args.verif:         "the verifier to be used",

        # Algorithm
        args.method:        "the algorithm to be used",
        args.max_it:        "max. number of iterations",
        args.rad:           "the distance restriction radius",
        args.delta:         "delta, percision parameter",
        
        # Domain
        args.dom_lb:        "the scalar of the domain's lower bound",
        args.dom_ub:        "the scalar of the domain's upper bound",
        
        # Interface
        args.no_out:        "no output, suppress exporting computed lb, ub as csvs",
        args.simple_res:    "simple results, outputing results as numbers in stdout",
        args.quiet:         "quiet, supress output",
        
        # Help
        args.help:          "help, print help"
    }
}

example_msg_header = "E.g.:"
example_msg = {
    # Required
    args.required: {
        args.x_star_path:   "<x_star_path>.csv",
        args.c_star:        "<c_star>",
        args.onnx_path:     "<onnx_description>.onnx"
    },

    # Optional
    args.optional: {
        # Input
        args.lb_path:       "<lb_path>.csv",
        args.ub_path:       "<ub_path>.csv",

        # Output
        args.out_dir:       "<output-subdir>",
        args.over_out:      None,
        args.save_images:   None,

        # Verifiers
        args.verif:         "<verif>",

        # Algorithm
        args.method:        "<algo>",
        args.max_it:        "<max_it>",
        args.rad:           "<rad>",
        args.delta:         "<delta>",
        
        # Domain
        args.dom_lb:        "<dom_lb>",
        args.dom_ub:        "<dom_ub>",
        
        # Interface
        args.no_out:        None,
        args.simple_res:    None,
        args.quiet:         None,
        
        # Help
        args.help:          None
    }
}


domain_msg_header = "Domain:"
domain_msg = {
    # Required
    args.required: {
        args.x_star_path:   "file",
        args.c_star:        "int",
        args.onnx_path:     "file"
    },

    # Optional
    args.optional: {
        # Input
        args.lb_path:       "file",
        args.ub_path:       "file",

        # Output
        args.out_dir:       "directory (will be created if not exists)",
        args.over_out:      None,
        args.save_images:   None,

        # Verifiers
        args.verif:         "(use " + args.cli_args[args.optional][args.help] + " " +\
                            args.help_args[args.help_verifs] + " to see the availabe options)",

        # Algorithm
        args.method:        "(use " + args.cli_args[args.optional][args.help] + " " +\
                            args.help_args[args.help_algos] + " to see the availabe options)",
        args.max_it:        "integer",
        args.rad:           "float",
        args.delta:         "float",
        
        # Domain
        args.dom_lb:        "float",
        args.dom_ub:        "float",
        
        # Interface
        args.no_out:        None,
        args.simple_res:    None,
        args.quiet:         None,
        
        # Help
        args.help:          args.help_args[args.help_algos]     + ": algorithms" + " " +\
                            args.help_args[args.help_path_conv] + ": path conventions."
    }
}


default_msg_header = "Default:"
default_msg = {
    # Required
    args.required: {
        args.x_star_path:   None,
        args.c_star:        None,
        args.onnx_path:     None
    },

    # Optional
    args.optional: {
        # Input
        args.lb_path:       None,
        args.ub_path:       None,

        # Output
        args.out_dir:       "outputs",
        args.over_out:      None,
        args.save_images:   None,

        # Verifiers
        args.verif:         "mara-sound",

        # Algorithm
        args.method:        "td",
        args.max_it:        "10_000",
        args.rad:           "1.0",
        args.delta:         "0.1",
        
        # Domain
        args.dom_lb:        "0.0",
        args.dom_ub:        "1.0",
        
        # Interface
        args.no_out:        None,
        args.simple_res:    None,
        args.quiet:         None,
        
        # Help
        args.help:          None
    }
}


def print_default_help():
    # Print help for required args
    print(help_required_header)
    for req_arg in args.cli_args[args.required].keys():
        print(f"{args.cli_args[args.required][req_arg]:<6}" + help_msg[args.required][req_arg])
        
        if example_msg[args.required][req_arg] != None:
            print(f"{'':<6}{example_msg_header + ' ' + example_msg[args.required][req_arg]}")
        
        if domain_msg[args.required][req_arg] != None:
            print(f"{'':<6}{domain_msg_header + ' ' + domain_msg[args.required][req_arg]}")
        
        if default_msg[args.required][req_arg] != None:
            print(f"{'':<6}{default_msg_header + ' ' + default_msg[args.required][req_arg]}")
        
        # give me some space
        print("")
    
    # give me some more space
    print("")

    # Print help for optional args
    print(help_optional_header)
    for opt_arg in args.cli_args[args.optional].keys():
        print(f"{args.cli_args[args.optional][opt_arg]:<6}{help_msg[args.optional][opt_arg]:<40}")
        
        if example_msg[args.optional][opt_arg] != None:
            print(f"{'':<6}{example_msg_header + ' ' + example_msg[args.optional][opt_arg]:<40}")
        
        if domain_msg[args.optional][opt_arg] != None:
            print(f"{'':<6}{domain_msg_header + ' ' + domain_msg[args.optional][opt_arg]:<40}")
        
        if default_msg[args.optional][opt_arg] != None:
            print(f"{'':<6}{default_msg_header + ' ' + default_msg[args.optional][opt_arg]:<40}")

        # give me some space
        print("")



#############
# Algo Help #
#############

## command line prefixes
help_algo_msg = {
    # Parallelepipedal Args 
    methods.bottom_up_linear_dfs:           "Bottom-Up Linear DFS",
    methods.bottom_up_dichotomic_dfs:       "Bottom-Up Dichotomic DFS",
    methods.bottom_up_bfs:                  "Bottom-Up BFS",
    methods.top_down:                       "Top-Down",
    
    # Cyclic Args
    methods.cyclic_bottom_up_linear:        "Cyclic Bottom-Up Linear",
    methods.cyclic_bottom_up_dichotomic:    "Cyclic Bottom-Up Dichotomic",
    methods.cyclic_top_down:                "Cyclic Top-Down",
    
    # Parallel + Parallel Composition
    methods.td_n_bu_l_dfs:                  "Top-Down + Bottom-Up Linear DFS",
    methods.td_n_bu_d_dfs:                  "Top-Down + Bottom-Up Dichotomic DFS",
    methods.td_n_bu_bfs:                    "Top-Down + Bottom-Up BFS",

    # Cyclic + Parallel Compositions
    methods.cbu_l_n_bu_l_dfs:               "Cyclic Bottom-Up Linear + Bottom-Up Linear DFS",
    methods.cbu_l_n_bu_d_dfs:               "Cyclic Bottom-Up Linear + Bottom-Up Dichotomic DFS",
    methods.cbu_l_n_bu_bfs:                 "Cyclic Bottom-Up Linear + Bottom-Up BFS",

    methods.cbu_d_n_bu_l_dfs:               "Cyclic Bottom-Up Dichotomic + Bottom-Up Linear DFS",
    methods.cbu_d_n_bu_d_dfs:               "Cyclic Bottom-Up Dichotomic + Bottom-Up Dichotomic DFS",
    methods.cbu_d_n_bu_bfs:                 "Cyclic Bottom-Up Dichotomic + Bottom-Up BFS",

    methods.ctd_n_bu_l_dfs:                 "Cyclic Top-Down + Bottom-Up Linear DFS",
    methods.ctd_n_bu_d_dfs:                 "Cyclic Top-Down + Bottom-Up Dichotomic DFS",
    methods.ctd_n_bu_bfs:                   "Cyclic Top-Down + Bottom-Up BFS",

    ## Methods for Complete Approximations
    methods.complete_bu:                    "Complete Bottom-Up"
}


def print_algo_help():
    print("Supported Algorithms:\n")
    for algo_arg in help_algo_msg.keys():
        print(f"{args.algo_args[algo_arg]:<20}{help_algo_msg[algo_arg]:<40}")



##################
# IO convensions #
##################

x_star_notes        = 0
output_def_notes    = 1
output_notes        = 2
notes               = 3

x_star_header       = 0
x_star_conv         = 1
x_star_conv_note_1  = 2
x_star_conv_note_2  = 3

output_def_header   = 4
output_def_conv     = 5
output_def_note_1   = 6
output_def_note_2   = 7

output_header       = 8
output_conv         = 9
output_note_1       = 10
output_note_2       = 11

note_header         = 12
note_1              = 13


x_star_header       = "# Input `" + args.cli_args[args.required][args.x_star_path] + "` Convension:"
output_def_header   = "# Output (Default) Convension:"
output_header       = "# Output `" + args.cli_args[args.optional][args.out_dir] + " <output_dir>` Convension:"
notes_header        = "# Notes:"

help_path_conv_msg = {
    x_star_notes: {
        x_star_conv:        "<path_header>/<dataset_folder>/inputs/<filename>.csv",
        x_star_conv_note_1: "1. The x_star vector must be included inside a directory named " +\
                            "'inputs'. The parent directory of the inputs dir is assumed " +\
                            "to be the dataset folder.",
        x_star_conv_note_2: "2. If the input does not adhere to this convension an error will" +\
                            "appear."
    },

    output_def_notes: {
        output_def_conv:    "<path_header>/<dataset_folder>/outputs/<algo-name>/<filename>_<lb | ub>.<csv | png>",
        output_def_note_1:  "1. The <algo-name> is specified the values of the -al argument.",
        output_def_note_2:  "2. The outputs and <algo_name> dirs will be created if not exist."
    },

    output_notes: {
        output_conv     :"<path_header>/<dataset_folder>/outputs/<output_dir>/<filename>_<lb | ub>.<csv | png>",
        output_note_1   :"1. Setting the outputs subdirectory, where the results will be stored.",
        output_note_2   :"2. The outputs and <output_dir> dirs will be created if not exist."
    },

    notes: {
        note_1:         "The results will *not* overwrite existing files. If this " +\
                        "behavior is desired use the `" + args.cli_args[args.optional][args.over_out] + "` argument."
    }
}

def print_io_convensions():
    print(x_star_header)
    for note in help_path_conv_msg[x_star_notes].keys():
        print(help_path_conv_msg[x_star_notes][note])
    print("")

    print(output_def_header)
    for note in help_path_conv_msg[output_def_notes].keys():
        print(help_path_conv_msg[output_def_notes][note])
    print("")

    print(output_header)
    for note in help_path_conv_msg[output_notes].keys():
        print(help_path_conv_msg[output_notes][note])
    print("")

    print(notes_header)
    for note in help_path_conv_msg[notes].keys():
        print(help_path_conv_msg[notes][note])



###############
# Help Verifs #
###############
## command line prefixes
help_verif_msg = {
    # Parallelepipedal Args 
    verifiers.marabou_sound:           "Marabou Sound Verifier",
    verifiers.marabou_complete:        "Marabou Complete Verifier"
}


def print_verif_help():
    print("Supported Verifiers:\n")
    for verif_arg in help_verif_msg.keys():
        print(f"{args.verif_args[verif_arg]:<20}{help_verif_msg[verif_arg]:<40}")

################
# Help Screens #
################
help_screens = {
    args.help_default:      print_default_help,
    args.help_algos:        print_algo_help,
    args.help_path_conv:    print_io_convensions,
    args.help_verifs:       print_verif_help
}