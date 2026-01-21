import cli.help as help
import cli.args as args
import cli.loader as loader
import cli.error_handling as errors
import cli.info as info

import cli.application as app

class Runner(loader.Loader):
    
    def __init__(self, argv):
        super().__init__(argv)

        self.application_run = None

    ## Help Screens
    def is_help_screen(self):
        return args.cli_args[args.optional][args.help] in self.argv

    def print_help_screen(self):
        info.print_header()
        help.help_screens[self[args.help]]()

    ## Algorithm Screens
    def is_run_algo(self):
        return not self[args.help] in self.argv
    
    def run_algo(self):
        self.application_run = app.Application(
            # required args
            self[args.x_star_path],
            self[args.c_star],
            self[args.onnx_path],
            self[args.out_dir],

            # optional args
            self[args.verif],
            self[args.method],
            self[args.max_it],
            self[args.rad],
            self[args.delta],
            self[args.dom_lb],
            self[args.dom_ub],
            self[args.timeout],
            not self[args.quiet],
            " ",
            self[args.lb_path],
            self[args.ub_path],
        )

        ## Header
        if not self[args.quiet]:
            info.print_header()
            self.application_run.print_input()
            self.application_run.print_setup()

        self.application_run.apply()

        ## Results
        if not self[args.quiet]:  self.application_run.print_results()
        if self[args.simple_res]: self.application_run.print_simple_results()


        ## save output
        if not self[args.no_out]:
            self.application_run.save_bounds(self[args.over_out])

        ## save images
        # no out overrides the save images argument if given
        if not self[args.no_out] and self[args.save_images]:
            self.application_run.image_save_bounds(self[args.over_out])



    ## Exec
    def exec(self):
        if self.is_help_screen():
            self.print_help_screen()
            exit(errors.error_all_ok)
        
        if self.is_run_algo():
            self.run_algo()
            exit(errors.error_all_ok)
        
        else: errors.print_error_message(errors.error_unkown_error)