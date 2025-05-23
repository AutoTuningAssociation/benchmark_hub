#!/usr/bin/env python
"""Point-in-Polygon host/device code tuner.

This program is used for auto-tuning the host and device code of a CUDA program
for computing the point-in-polygon problem for very large datasets and large
polygons.

The time measurements used as a basis for tuning include the time spent on
data transfers between host and device memory. The host code uses device mapped
host memory to overlap communication between host and device with kernel
execution on the GPU. Because each input is read only once and each output
is written only once, this implementation almost fully overlaps all
communication and the kernel execution time dominates the total execution time.

The code has the option to precompute all polygon line slopes on the CPU and
reuse those results on the GPU, instead of recomputing them on the GPU all
the time. The time spent on precomputing these values on the CPU is also
taken into account by the time measurement in the code.

This code was written for use with the Kernel Tuner. See:
     https://github.com/benvanwerkhoven/kernel_tuner

Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
"""

import sys
from collections import OrderedDict

import kernel_tuner
import numpy as np
from kernel_tuner.file_utils import store_metadata_file, store_output_file

file_path_results = "../last_run/_tune_configuration-results.json"
file_path_metadata = "../last_run/_tune_configuration-metadata.json"


def tune(device_name: str, strategy="mls", strategy_options=None, verbose=True, quiet=False, simulation_mode=True):
    # set the number of points and the number of vertices
    size = np.int32(2e7)
    problem_size = size
    args = []

    # setup tunable parameters
    tune_params = OrderedDict()
    tune_params["between_method"] = [0, 1, 2, 3]
    tune_params["block_size_x"] = [32 * i for i in range(1, 32)]  # multiple of 32
    tune_params["tile_size"] = [1] + [2 * i for i in range(1, 11)]
    tune_params["use_method"] = [0, 1, 2]

    # tell the Kernel Tuner how to compute the grid dimensions from the problem_size
    grid_div_x = ["block_size_x", "tile_size"]

    metrics = OrderedDict()
    metrics["MPoints/s"] = lambda p: (size / 1e6) / (p["time"] / 1e3)

    # start tuning
    results, env = kernel_tuner.tune_kernel(
        "cn_pnpoly",
        "pnpoly.cu",
        problem_size,
        args,
        tune_params,
        grid_div_x=grid_div_x,
        lang="C",
        cache="../cachefiles/pnpoly/" + device_name.lower(),
        metrics=metrics,
        verbose=verbose,
        quiet=quiet,
        strategy=strategy,
        strategy_options=strategy_options,
        simulation_mode=simulation_mode,
    )

    store_output_file(file_path_results, results, tune_params)
    store_metadata_file(file_path_metadata)
    return results, env


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./pnpoly.py [device name]")
        exit(1)
    device_name = sys.argv[1]

    tune(device_name)
