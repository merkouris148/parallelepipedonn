import typing

import cli.methods as methods
import cli.verifiers as verifiers

#########################################
# Mapping Method CLI args to Method Ids # 
#########################################

## command line prefixes
algo_args = {
    # Parallelepipedal Args 
    methods.bottom_up_linear_dfs:           "bu-l-dfs",
    methods.bottom_up_dichotomic_dfs:       "bu-d-dfs",
    methods.bottom_up_bfs:                  "bu-bfs",
    methods.top_down:                       "td",
    
    # Cyclic Args
    methods.cyclic_bottom_up_linear:        "c-bu-l",
    methods.cyclic_bottom_up_dichotomic:    "c-bu-d",
    methods.cyclic_top_down:                "c-td",
    
    # Parallel + Parallel Composition
    methods.td_n_bu_l_dfs:                  "td+bu-l-dfs",
    methods.td_n_bu_d_dfs:                  "td+bu-d-dfs",
    methods.td_n_bu_bfs:                    "td+bu-bfs",

    # Cyclic + Parallel Compositions
    methods.cbu_l_n_bu_l_dfs:               "c-bu-l+bu-l-dfs",
    methods.cbu_l_n_bu_d_dfs:               "c-bu-l+bu-d-dfs",
    methods.cbu_l_n_bu_bfs:                 "c-bu-l+bu-bfs",

    methods.cbu_d_n_bu_l_dfs:               "c-bu-d+bu-l-dfs",
    methods.cbu_d_n_bu_d_dfs:               "c-bu-d+bu-d-dfs",
    methods.cbu_d_n_bu_bfs:                 "c-bu-d+bu-bfs",

    methods.ctd_n_bu_l_dfs:                 "c-td+bu-l-dfs",
    methods.ctd_n_bu_d_dfs:                 "c-td+bu-d-dfs",
    methods.ctd_n_bu_bfs:                   "c-td+bu-bfs",

    ## Methods for Complete Approximations
    methods.complete_bu:                    "complete-bu"
}

args_algo = {
    # Parallelepipedal Args 
    algo_args[methods.bottom_up_linear_dfs]:        methods.bottom_up_linear_dfs,
    algo_args[methods.bottom_up_dichotomic_dfs]:    methods.bottom_up_dichotomic_dfs,
    algo_args[methods.bottom_up_bfs]:               methods.bottom_up_bfs,
    algo_args[methods.top_down]:                    methods.top_down,
    
    # Cyclic Args
    algo_args[methods.cyclic_bottom_up_linear]:     methods.cyclic_bottom_up_linear,
    algo_args[methods.cyclic_bottom_up_dichotomic]: methods.cyclic_bottom_up_dichotomic,
    algo_args[methods.cyclic_top_down]:             methods.cyclic_top_down,
    
    # Parallel + Parallel Composition
    algo_args[methods.td_n_bu_l_dfs]:               methods.td_n_bu_l_dfs,
    algo_args[methods.td_n_bu_d_dfs]:               methods.td_n_bu_d_dfs,
    algo_args[methods.td_n_bu_bfs]:                 methods.td_n_bu_bfs,

    # Cyclic + Parallel Compositions
    algo_args[methods.cbu_l_n_bu_l_dfs]:            methods.cbu_l_n_bu_l_dfs,
    algo_args[methods.cbu_l_n_bu_d_dfs]:            methods.cbu_l_n_bu_d_dfs,
    algo_args[methods.cbu_l_n_bu_bfs]:              methods.cbu_l_n_bu_bfs,

    algo_args[methods.cbu_d_n_bu_l_dfs]:            methods.cbu_d_n_bu_l_dfs,
    algo_args[methods.cbu_d_n_bu_d_dfs]:            methods.cbu_d_n_bu_d_dfs,
    algo_args[methods.cbu_d_n_bu_bfs]:              methods.cbu_d_n_bu_bfs,

    algo_args[methods.ctd_n_bu_l_dfs]:              methods.ctd_n_bu_l_dfs,
    algo_args[methods.ctd_n_bu_d_dfs]:              methods.ctd_n_bu_d_dfs,
    algo_args[methods.ctd_n_bu_bfs]:                methods.ctd_n_bu_bfs,

    ## Methods for Complete Approximations
    algo_args[methods.complete_bu]:                 methods.complete_bu
}



verif_args = {
    verifiers.marabou_sound:    "mara-sound",
    verifiers.marabou_complete: "mara-complete"
}

args_verif = {
    verif_args[verifiers.marabou_sound]:    verifiers.marabou_sound,
    verif_args[verifiers.marabou_complete]: verifiers.marabou_complete
}



############
# CLI Args #
############

required = 0
optional = 1

# Required
x_star_path    = 0
c_star         = 1
onnx_path      = 2

# Optional
lb_path        = 3
ub_path        = 4
out_dir        = 5
over_out       = 6
save_images    = 7
method         = 8
max_it         = 9
rad            = 10
delta          = 11
dom_ub         = 12
dom_lb         = 13
no_out         = 14
simple_res     = 15
quiet          = 16
help           = 17
verif          = 18


cli_args = {
    
    ## Required Args
    required: {
        x_star_path:    "-x",
        c_star:         "-c",
        onnx_path:      "-nn"
    },

    ## Optional Args
    optional: {
        # Input
        lb_path:        "-lb",
        ub_path:        "-ub",
        
        # Output
        out_dir:        "-od",
        over_out:       "-ov",
        save_images:    "-si",

        # Algorithm
        method:         "-al",
        max_it:         "-mi",
        rad:            "-r",
        delta:          "-d",
        dom_ub:         "-du",
        dom_lb:         "-dl",

        # Verifier
        verif:          "-v",
        
        # Interface
        no_out:         "-no",
        simple_res:     "-sr",
        quiet:          "-q",
        help:           "-h",
    }
}

def is_arg(argv: typing.List[str], index: int) -> bool:
    assert 0 <= index and index <= len(argv)

    return argv[index][0] == "-"


#############
# Help Args #
#############

help_no         = 0
help_default    = 1
help_path_conv  = 2
help_algos      = 3
help_verifs     = 4

help_args = {
    help_path_conv: "pc",
    help_algos:     "al",
    help_verifs:    "v"
}

args_help = {
    help_args[help_path_conv]:  help_path_conv,
    help_args[help_algos]:      help_algos,
    help_args[help_verifs]:     help_verifs
}


defaults = {
    
    ## Required Args
    required: {
        x_star_path:    None,
        c_star:         None,
        onnx_path:      None
    },

    ## Optional Args
    optional: {
        # Input
        lb_path:        "",
        ub_path:        "",

        # Output
        out_dir:        "outputs",
        over_out:       False,
        save_images:    False,

        # Verifiers
        verif:          verifiers.marabou_sound,

        # Algorithm
        method:         methods.top_down,
        max_it:         10_000,
        rad:            1.0,
        delta:          0.1,
        dom_lb:         0,
        dom_ub:         1,

        # Interface
        no_out:         False,
        simple_res:     False,
        quiet:          False,
        help:           help_no
    }
}