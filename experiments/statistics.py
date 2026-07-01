## Python Libraries
import typing as t
import os
import re
from datetime import datetime

## 3rd Party Libraries
import numpy as np

## Custom Libraries
import sys
sys.path.append("..")
import guarantees.parallelepipedal as guarantees
import geometry.interval as interval


#########
# RegEx #
#########

## x_star pattern
x_star_pattern = re.compile("[0-9]*-[0-9]*.csv")
is_x_star_file = lambda filename: x_star_pattern.match(filename) is not None

## bounds pattern
lb_pattern = re.compile("[0-9]*-[0-9]*_lb.csv")
is_lb_file = lambda filename: lb_pattern.match(filename) is not None

ub_pattern = re.compile("[0-9]*-[0-9]*_ub.csv")
is_ub_file = lambda filename: ub_pattern.match(filename) is not None

#
# variance
single_variance     = lambda data_point, avg: (data_point - avg)**2
def multiple_variances(data_points, avg):
    avgs = [avg] * len(data_points)

    return sum(
        map(
            single_variance,
            data_points,
            avgs
        )
    ) / len(data_points)

#########
# Class #
#########

class Statistics:
    def __init__(
        self,
        inputs_dir: str,
        bounds_dir: str,
        domain: interval.Interval,
        output_dir: t.Optional[str] = None
    ) -> None:
        ## Checks
        assert os.path.isdir(inputs_dir)
        assert os.path.isdir(bounds_dir)
        assert not domain.empty()

        ## Initialize
        self.inputs_dir = inputs_dir
        self.bounds_dir = bounds_dir
        self.domain     = domain
        self.output_dir = output_dir if output_dir is not None else bounds_dir

        ## mkdirs
        os.makedirs(self.output_dir, exist_ok=True)

        ## load guarantees
        self.guarantees: t.List[guarantees.ParallelepipedalGuarantee] = []
        self._load_guarantees()
        assert len(self.guarantees) > 0

        ## data
        self.complexities:  t.List[int]     = []
        self.min_edge_lens: t.List[float]   = []
        self.avg_lens:      t.List[float]   = []
        self.diams:         t.List[float]   = []
        self.apothems:      t.List[float]   = []
        self.perimeters:    t.List[float]   = []
        self._load_data()

        ## Statistics
        # complexity
        self.min_comp           = min(self.complexities)
        self.avg_comp           = sum(self.complexities) / len(self.complexities)
        self.max_comp           = max(self.complexities)
        self.var_comp           = multiple_variances(self.complexities, self.avg_comp)
        self.std_dev_comp       = np.sqrt(self.var_comp)

        # min edge length
        self.min_min_edge_len   = min(self.min_edge_lens)
        self.avg_min_edge_len   = sum(self.min_edge_lens) / len(self.min_edge_lens)
        self.max_min_edge_len   = max(self.min_edge_lens)
        self.var_min_edge_len   = multiple_variances(self.min_edge_lens, self.avg_min_edge_len)
        self.std_dev_min_edge_len = np.sqrt(self.var_min_edge_len)

        # avg. edge len.
        self.min_avg_edge_len   = min(self.avg_lens)
        self.avg_avg_edge_len   = sum(self.avg_lens) / len(self.avg_lens)
        self.max_avg_edge_len   = max(self.avg_lens)
        self.var_avg_edge_len   = multiple_variances(self.avg_lens, self.avg_avg_edge_len)
        self.std_dev_avg_edge_len = np.sqrt(self.var_avg_edge_len)

        # diameter (max. edge len.)
        self.min_diam           = min(self.diams)
        self.avg_diam           = sum(self.diams) / len(self.diams)
        self.max_diam           = max(self.diams)
        self.var_diam           = multiple_variances(self.diams, self.avg_diam)
        self.std_dev_diam       = np.sqrt(self.var_diam)

        # perimeter
        self.min_perimeter    = min(self.perimeters)
        self.avg_perimeter    = sum(self.perimeters) / len(self.perimeters)
        self.max_perimeter    = max(self.perimeters)
        self.var_perimeter    = multiple_variances(self.perimeters, self.avg_perimeter)
        self.std_dev_perimeter = np.sqrt(self.var_perimeter)

        # apothem
        self.min_apothem    = min(self.apothems)
        self.avg_apothem    = sum(self.apothems) / len(self.apothems)
        self.max_apothem    = max(self.apothems)
        self.var_apothem    = multiple_variances(self.apothems, self.avg_apothem)
        self.std_dev_apothem = np.sqrt(self.var_apothem)
        

    
    def _load_guarantees(self):
        # lower bounds
        lb_csvs     = [
            self.bounds_dir + "/" + item
            for item in os.listdir(self.bounds_dir)
            if
                os.path.isfile(self.bounds_dir + "/" + item)
            and is_lb_file(item)
        ]
        lb_csvs.sort()

        ## Compute Available Lower Bounds
        lb_available_data = [
            item 
            for item in os.listdir(self.bounds_dir)
            if
                os.path.isfile(self.bounds_dir + "/" + item)
            and is_lb_file(item)
        ]
        lb_available_data = list(map(lambda name: name.split("_")[0], lb_available_data))
        lb_available_data.sort()
        #print(lb_available_data)

        # upper bounds
        #print(list(os.listdir(self.bounds_dir)))
        ub_csvs     = [
            self.bounds_dir + "/" + item
            for item in os.listdir(self.bounds_dir)
            if
                os.path.isfile(self.bounds_dir + "/" + item)
            and is_ub_file(item)
        ]
        ub_csvs.sort()

        ## Compute Available Upper Bounds
        ub_available_data = [
            item 
            for item in os.listdir(self.bounds_dir)
            if
                os.path.isfile(self.bounds_dir + "/" + item)
            and is_ub_file(item)
        ]
        ub_available_data = list(map(lambda name: name.split("_")[0], ub_available_data))
        ub_available_data.sort()
        #print(ub_available_data)

        ## checks
        assert len(lb_csvs) == len(ub_csvs)
        assert lb_available_data == ub_available_data

        # due to the timeouts the lb_csvs do not match the x_stars
        # x_stars
        x_star_csvs = [
            self.inputs_dir + "/" + item
            for item in os.listdir(self.inputs_dir)
            if
                os.path.isfile(self.inputs_dir + "/" + item)
            and is_x_star_file(item)
            and item.split(".")[0] in lb_available_data
        ]
        x_star_csvs.sort()

        ## checks
        assert len(x_star_csvs) == len(lb_available_data)

        ## load guarantees
        n           = len(x_star_csvs)
        dummy_delta = -1.0
        dummy_class = -1
        for i in range(n):
            x_star  = np.genfromtxt(x_star_csvs[i])
            lb      = np.genfromtxt(lb_csvs[i])
            ub      = np.genfromtxt(ub_csvs[i])
            guarantee = guarantees.ParallelepipedalGuarantee(
                x_star,
                dummy_class,
                dummy_delta,
                self.domain
            )
            guarantee.set_bounds(lb, ub)

            self.guarantees.append(guarantee)


    def _load_data(self):
        for guarantee in self.guarantees:
            self.complexities.append(
                guarantee.calc_complexity()
            )
            self.min_edge_lens.append(
                guarantee.min_edge_len()
            )
            self.avg_lens.append(
                guarantee.avg_edge_len()
            )
            self.diams.append(
                guarantee.diam()
            )
            self.apothems.append(
                guarantee.apothem()
            )
            self.perimeters.append(
                guarantee.perimeter()
            )

    def __str__(self):
        return "### Experimental Reults ###"                                    + "\n" +\
            f"{'Date-Time:':<26}"           + str(datetime.now())               + "\n" +\
            "# Description Complexity:"                                         + "\n" +\
            f"{'Min.:':<26}"                + str(self.min_comp)                + "\n" +\
            f"{'Avg.:':<26}"                + str(self.avg_comp)                + "\n" +\
            f"{'Max.:':<26}"                + str(self.max_comp)                + "\n" +\
            f"{'Var.:':<26}"                + str(self.var_comp)                + "\n" +\
            f"{'Std Dev.:':<26}"            + str(self.std_dev_comp)            + "\n" +\
            "-" * 60                                                            + "\n" +\
            "# Minimum Edge Length:"                                            + "\n" +\
            f"{'Min.:':<26}"                + str(self.min_min_edge_len)        + "\n" +\
            f"{'Avg.:':<26}"                + str(self.avg_min_edge_len)        + "\n" +\
            f"{'Max.:':<26}"                + str(self.max_min_edge_len)        + "\n" +\
            f"{'Var.:':<26}"                + str(self.var_min_edge_len)        + "\n" +\
            f"{'Std Dev.:':<26}"            + str(self.std_dev_min_edge_len)    + "\n" +\
            "-" * 60                                                            + "\n" +\
            "# Diameter:"                                                       + "\n" +\
            f"{'Min.:':<26}"                + str(self.min_diam)                + "\n" +\
            f"{'Avg.:':<26}"                + str(self.avg_diam)                + "\n" +\
            f"{'Max.:':<26}"                + str(self.max_diam)                + "\n" +\
            f"{'Var.:':<26}"                + str(self.var_diam)                + "\n" +\
            f"{'Std Dev.:':<26}"            + str(self.std_dev_diam)            + "\n" +\
            "-" * 60                                                            + "\n" +\
            "# Avg. Edge Length:"                                               + "\n" +\
            f"{'Min.:':<26}"                + str(self.min_avg_edge_len)        + "\n" +\
            f"{'Avg.:':<26}"                + str(self.avg_avg_edge_len)        + "\n" +\
            f"{'Max.:':<26}"                + str(self.max_avg_edge_len)        + "\n" +\
            f"{'Var.:':<26}"                + str(self.var_avg_edge_len)        + "\n" +\
            f"{'Std Dev.:':<26}"            + str(self.std_dev_avg_edge_len)    + "\n" +\
            "-" * 60                                                            + "\n" +\
            "# Perimeter:"                                                      + "\n" +\
            f"{'Min.:':<26}"                + str(self.min_perimeter)           + "\n" +\
            f"{'Avg.:':<26}"                + str(self.avg_perimeter)           + "\n" +\
            f"{'Max.:':<26}"                + str(self.max_perimeter)           + "\n" +\
            f"{'Var.:':<26}"                + str(self.var_perimeter)           + "\n" +\
            f"{'Std Dev.:':<26}"            + str(self.std_dev_perimeter)       + "\n" +\
            "-" * 60                                                            + "\n" +\
            "# Apothem:"                                                        + "\n" +\
            f"{'Min.:':<26}"                + str(self.min_apothem)             + "\n" +\
            f"{'Avg.:':<26}"                + str(self.avg_apothem)             + "\n" +\
            f"{'Max.:':<26}"                + str(self.max_apothem)             + "\n" +\
            f"{'Var.:':<26}"                + str(self.var_apothem)             + "\n" +\
            f"{'Std Dev.:':<26}"            + str(self.std_dev_apothem)         + "\n" +\
            "-" * 60
    
    def save(self):
        output_file = self.output_dir
        if output_file[-1] != "/": output_file += "/"
        output_file += "adhoc_results.txt"

        f = open(output_file, "w")
        f.write(str(self))
        f.close()



########
# Main #
########
domain_28_28_grayscale = interval.Interval(
    np.zeros((28, 28)),
    np.ones((28, 28))
)

mnist_ds    = 0
fmnist_ds   = 1

datasets    = [
    mnist_ds,
    fmnist_ds
]

ds_domains = [
    domain_28_28_grayscale,
    domain_28_28_grayscale
]

input_paths = [
    "../data/inputs/MNIST",
    "../data/inputs/FashionMNIST",
]

bound_paths = [
    [
        "../data/outputs/MNIST/td-09-06-2026 (master)",
        "../data/outputs/MNIST/complete-bu-10-06-2026 (master)",
        "../data/outputs/MNIST/c-td-11-06-2026 (master)",
        "../data/outputs/MNIST/complete-c-d-bu-11-06-2026 (master)",
    ],
    [
        "../data/outputs/FashionMNIST/td-10-06-2026 (master)",
        "../data/outputs/FashionMNIST/complete-bu-10-06-2026 (master)",
        "../data/outputs/FashionMNIST/c-td-12-06-2026 (master)",
        "../data/outputs/FashionMNIST/complete-c-d-bu-12-06-2026 (master)",
    ]
]

verbose = True

if __name__=="__main__":
    for dataset in datasets:
        input_path = input_paths[dataset]
        if verbose: print("Reading Inputs from:", input_path)

        for bound_path in bound_paths[dataset]:
            if verbose: print()
            if verbose: print("  > Reading Bounds from:", bound_path)
            stats = Statistics(
                input_path,
                bound_path,
                ds_domains[dataset]
            )
            if verbose: print("  > Writing Results in:", stats.output_dir)
            stats.save()
            if verbose: print()
