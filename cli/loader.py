## Typing
import typing


## Custom
import cli.error_handling as errors
import cli.args as args
import cli.checks as checks
import cli.loads as loads


class Loader(dict):

    def __init__(self, argv: typing.List[str]) -> None:
        
        self.argv = argv

        ## Initialize Defaults
        for req_args in args.cli_args[args.required].keys():
            self[req_args] = args.defaults[args.required][req_args]

        for opt_args in args.cli_args[args.optional].keys():
            self[opt_args] = args.defaults[args.optional][opt_args]
        


        ## Check Arguments
        for req_args in args.cli_args[args.required].keys():
            error, code = checks.check_argument[args.required][req_args](argv)
            if error: errors.print_error_message(code)

        for opt_args in args.cli_args[args.optional].keys():
            error, code = checks.check_argument[args.optional][opt_args](argv)
            if error: errors.print_error_message(code)
        


        ## Load Values
        for req_args in args.cli_args[args.required].keys():
            req_val = loads.load[args.required][req_args](argv)
            if req_val != None: self[req_args] = req_val

        for opt_args in args.cli_args[args.optional].keys():
            opt_val = loads.load[args.optional][opt_args](argv)
            if opt_val != None: self[opt_args] = opt_val
