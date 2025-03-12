#!/usr/bin/env python
import os
import sys
from collections import OrderedDict
import time

import numpy

import kernel_tuner
from kernel_tuner.file_utils import store_metadata_file, store_output_file


def ops(w, h, fw, fh):
    return (w * h * fw * fh * 2) / 1e9


unit = "GFLOP"
w = h = 4096
fw = fh = 15
inputs = [w, h, fw, fh]
total_flops = ops(w, h, fw, fh)


# def tune(inputs, lang, strategy):
def tune(
    device_name: str,
    strategy="brute_force",
    strategy_options=None,
    verbose=True,
    quiet=False,
    simulation_mode=False,
    lang="CUDA",
):
    if lang == "CUDA":
        kernel_file = "/convolution_milo.cu"
    elif lang == "HIP":
        kernel_file = "/convolution_milo.cu.hip"

    with open(os.path.dirname(os.path.realpath(__file__)) + kernel_file, "r") as f:
        kernel_string = f.read()

    # setup tunable parameters
    tune_params = OrderedDict()

    # tune_params["pwr_limit"] = get_pwr_limit(pwr_limit, 0)

    image_width, image_height, filter_width, filter_height = inputs

    tune_params["block_size_x"] = [16 * i for i in range(1, 17)]
    tune_params["block_size_y"] = [2**i for i in range(5)]
    tune_params["tile_size_x"] = [i for i in range(1, 5)]
    tune_params["tile_size_y"] = [i for i in range(1, 5)]
    tune_params["read_only"] = [0, 1]  # toggle using the read-only cache

    # do dry run
    # tune_params["nvml_gr_clock"] = [2100]
    # tune_params["block_size_x"] = [16]
    # tune_params["block_size_y"] = [1]
    # tune_params["tile_size_x"] = [1, 2, 4]
    # tune_params["tile_size_y"] = [1]
    # tune_params["read_only"] = [1]    #toggle using the read-only cache

    tune_params["use_padding"] = [0, 1]  # toggle the insertion of padding in shared memory

    # limit the search to only use padding when its effective
    restrict = [
        "(use_padding==0 or (block_size_x % 32 != 0))",
        "((block_size_x*tile_size_x+4)*(block_size_y*tile_size_y+4) < 12*1024)",
    ]
    restrict.append(
        "(((block_size_x*tile_size_x+%d)*(block_size_y*tile_size_y+%d)) < 12*1024)"
        % (filter_width - 1, filter_height - 1)
    )
    restrict.append("block_size_x * block_size_y <= 1024")

    # print(restrict)

    problem_size = (image_width, image_height)
    size = numpy.prod(problem_size)
    largest_fh = filter_height
    largest_fw = filter_width
    input_size = (problem_size[0] + largest_fw - 1) * (problem_size[1] + largest_fh - 1)

    output_image = numpy.zeros(size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(largest_fh * largest_fw).astype(numpy.float32)

    cmem_args = {"d_filter": filter_weights}
    args = [output_image, input_image, filter_weights]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    total_flops = ops(*inputs)
    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)

    base_cachepath = f"../cachefiles/convolution_milo/{device_name.upper()}"

    # start tuning
    start = time.time()
    results, env = kernel_tuner.tune_kernel(
        "convolution_kernel",
        kernel_string,
        problem_size,
        args,
        tune_params,
        grid_div_y=grid_div_y,
        grid_div_x=grid_div_x,
        cmem_args=cmem_args,
        restrictions=restrict,
        cache=base_cachepath,
        metrics=metrics,
        lang=lang,
        iterations=32,
        device=0,
        verbose=verbose,
        quiet=quiet,
        strategy=strategy,
        strategy_options=strategy_options,
        simulation_mode=simulation_mode,
    )
    end = time.time()
    env["execution_time"] = end - start

    store_output_file(f"{base_cachepath}-results.json", results, tune_params)
    store_metadata_file(f"{base_cachepath}-metadata.json")
    return results, env


if __name__ == "__main__":
    language = sys.argv[1]
    device_name = sys.argv[2]

    if len(sys.argv) != 3:
        raise ValueError(f"Usage: python convolution_milo.py [language ('HIP' or 'CUDA')] [device name], given: {sys.argv}")

    if language not in ("HIP", "CUDA"):
        raise ValueError(f"{language} not valid, specify HIP or CUDA")

    tune(device_name=device_name, lang=language)

# # list of configurations
# configurations = [
#     {
#         "strategy": "dual_annealing",
#         "strategy_options": {"method": "COBYLA"},
#         "filename": f"convolution{lang}_DA_COBYLA.json",
#     },
#     {
#         "strategy": "dual_annealing",
#         "strategy_options": {"method": "L-BFGS-B"},
#         "filename": f"convolution{lang}_DA_L-BFGS-B.json",
#     },
#     {
#         "strategy": "dual_annealing",
#         "strategy_options": {"method": "SLSQP"},
#         "filename": f"convolution{lang}_DA_SLSQP.json",
#     },
#     {
#         "strategy": "dual_annealing",
#         "strategy_options": {"method": "CG"},
#         "filename": f"convolution{lang}_DA_CG.json",
#     },
#     {
#         "strategy": "dual_annealing",
#         "strategy_options": {"method": "Powell"},
#         "filename": f"convolution{lang}_DA_Powell.json",
#     },
#     {
#         "strategy": "dual_annealing",
#         "strategy_options": {"method": "Nelder-Mead"},
#         "filename": f"convolution{lang}_DA_Nelder-Mead.json",
#     },
#     {
#         "strategy": "dual_annealing",
#         "strategy_options": {"method": "BFGS"},
#         "filename": f"convolution{lang}_DA_BFGS.json",
#     },
#     {
#         "strategy": "dual_annealing",
#         "strategy_options": {"method": "trust-constr"},
#         "filename": f"convolution{lang}_DA_trust-constr.json",
#     },
#     {"strategy": "mls", "strategy_options": None, "filename": f"convolution{lang}_MLS.json"},
#     {"strategy": "greedy_ils", "strategy_options": None, "filename": f"convolution{lang}_ILS.json"},
#     {
#         "strategy": "basinhopping",
#         "strategy_options": {"method": "Nelder-Mead"},
#         "filename": f"convolution{lang}_BN_Nelder-Mead.json",
#     },
#     {
#         "strategy": "basinhopping",
#         "strategy_options": {"method": "Powell"},
#         "filename": f"convolution{lang}_BN_Powell.json",
#     },
#     {"strategy": "basinhopping", "strategy_options": {"method": "CG"}, "filename": f"convolution{lang}_BN_CG.json"},
#     {
#         "strategy": "basinhopping",
#         "strategy_options": {"method": "BFGS"},
#         "filename": f"convolution{lang}_BN_BFGS.json",
#     },
#     {
#         "strategy": "basinhopping",
#         "strategy_options": {"method": "L-BFGS-B"},
#         "filename": f"convolution{lang}_BN_L-BFGS-B.json",
#     },
#     # {"strategy": "basinhopping", "strategy_options": {"method": "TNC"}, "filename": f'convolution{lang}_BN_TNC.json'},
#     {
#         "strategy": "basinhopping",
#         "strategy_options": {"method": "COBYLA"},
#         "filename": f"convolution{lang}_BN_COBYLA.json",
#     },
#     {
#         "strategy": "minimize",
#         "strategy_options": {"method": "Nelder-Mead"},
#         "filename": f"convolution{lang}_M_Nelder-Mead.json",
#     },
#     {
#         "strategy": "minimize",
#         "strategy_options": {"method": "Powell"},
#         "filename": f"convolution{lang}_M_Powell.json",
#     },
#     {"strategy": "minimize", "strategy_options": {"method": "CG"}, "filename": f"convolution{lang}_M_CG.json"},
#     {"strategy": "minimize", "strategy_options": {"method": "BFGS"}, "filename": f"convolution{lang}_M_BFGS.json"},
#     {
#         "strategy": "minimize",
#         "strategy_options": {"method": "L-BFGS-B"},
#         "filename": f"convolution{lang}_M_L-BFGS-B.json",
#     },
#     {"strategy": "minimize", "strategy_options": {"method": "TNC"}, "filename": f"convolution{lang}_M_TNC.json"},
#     {
#         "strategy": "minimize",
#         "strategy_options": {"method": "COBYLA"},
#         "filename": f"convolution{lang}_M_COBYLA.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "best1bin"},
#         "filename": f"convolution{lang}_DE_best1bin.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "best1exp"},
#         "filename": f"convolution{lang}_DE_best1exp.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "rand1exp"},
#         "filename": f"convolution{lang}_DE_rand1exp.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "randtobest1exp"},
#         "filename": f"convolution{lang}_DE_randtobest1exp.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "best2exp"},
#         "filename": f"convolution{lang}_DE_best2exp.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "rand2exp"},
#         "filename": f"convolution{lang}_DE_rand2exp.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "randtobest1bin"},
#         "filename": f"convolution{lang}_DE_randtobest1bin.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "best2bin"},
#         "filename": f"convolution{lang}_DE_best2bin.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "rand2bin"},
#         "filename": f"convolution{lang}_DE_rand2bin.json",
#     },
#     {
#         "strategy": "diff_evo",
#         "strategy_options": {"method": "rand1bin"},
#         "filename": f"convolution{lang}_DE_rand1bin.json",
#     },
#     {"strategy": "genetic_algorithm", "strategy_options": None, "filename": f"convolution{lang}_GE.json"},
#     {"strategy": "pso", "strategy_options": None, "filename": f"convolution{lang}_PSO.json"},
#     {"strategy": "firefly_algorithm", "strategy_options": None, "filename": f"convolution{lang}_FA.json"},
#     {"strategy": "simulated_annealing", "strategy_options": None, "filename": f"convolution{lang}_SA.json"},
# ]
