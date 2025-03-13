#!/usr/bin/env python
import os
import sys
from collections import OrderedDict
import time

import numpy
import numpy as np

import kernel_tuner
from kernel_tuner.file_utils import store_metadata_file, store_output_file

max_threads = 1024


def test_temporal_tiling():
    problem_size = (7, 7)

    tune_params, max_tfactor = get_tunable_parameters(problem_size)

    temp_src, power, temp_dst = get_input_data(problem_size, max_tfactor)

    test_input = np.array([i for i in range(np.prod(problem_size))]).reshape(*problem_size)
    temp_src[max_tfactor:-max_tfactor, max_tfactor:-max_tfactor] = test_input

    # setup arguments
    step_div_cap, Rx_1, Ry_1, Rz_1 = get_input_arguments(*problem_size)
    args = [power, temp_src, temp_dst, Rx_1, Ry_1, Rz_1, step_div_cap]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    # call the kernel once with temporal tiling factor = 1
    params = dict(
        grid_width=problem_size[0],
        grid_height=problem_size[1],
        block_size_x=16,
        block_size_y=16,
        tile_size_x=1,
        tile_size_y=1,
        temporal_tiling_factor=1,
        max_tfactor=max_tfactor,
    )
    reference = kt.run_kernel(
        "calculate_temp",
        "hotspot_milo.cu.hip",
        problem_size,
        args,
        params,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
    )

    print(reference[2])

    # replace the input with the output of the first kernel
    temp2 = np.zeros_like(temp_src)
    temp2[max_tfactor:-max_tfactor, max_tfactor:-max_tfactor] = reference[2]

    # call the kernel again with temporal tiling factor = 1
    args2 = [power, temp2, temp_dst, Rx_1, Ry_1, Rz_1, step_div_cap]
    reference2 = kt.run_kernel(
        "calculate_temp",
        "hotspot_milo.cu.hip",
        problem_size,
        args2,
        params,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
    )
    answer = [None for _ in args]
    answer[2] = reference2[2]

    print(answer[2])

    # tune the kernel with temporal tiling factor = 2
    tune_params["block_size_y"] = [16, 32]
    tune_params["block_size_x"] = [16, 32]
    tune_params["tile_size_x"] = [1, 2, 4]
    tune_params["tile_size_y"] = [1, 2, 4]
    tune_params["temporal_tiling_factor"] = [2]

    results, env = kt.tune_kernel(
        "calculate_temp",
        "hotspot_milo.cu.hip",
        problem_size,
        args,
        tune_params,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        verbose=True,
        answer=answer,
    )


def get_input_arguments(grid_rows, grid_cols):
    # maximum power density possible (say 300W for a 10mm x 10mm chip)  */
    MAX_PD = 3.0e6

    # required precision in degrees */
    PRECISION = 0.001
    SPEC_HEAT_SI = 1.75e6
    K_SI = 100

    # capacitance fitting factor    */
    FACTOR_CHIP = 0.5

    # chip parameters   */
    t_chip = 0.0005
    chip_height = 0.016
    chip_width = 0.016

    grid_height = chip_height / grid_rows
    grid_width = chip_width / grid_cols
    cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height
    Rx = grid_width / (2.0 * K_SI * t_chip * grid_height)
    Ry = grid_height / (2.0 * K_SI * t_chip * grid_width)
    Rz = t_chip / (K_SI * grid_height * grid_width)
    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    step = PRECISION / max_slope
    step_div_cap = step / cap

    return [np.float32(i) for i in [step_div_cap, 1 / Rx, 1 / Ry, 1 / Rz]]


def get_tunable_parameters(problem_size):
    tune_params = OrderedDict()

    # input sizes need at compile time
    tune_params["grid_width"] = [problem_size[0]]
    tune_params["grid_height"] = [problem_size[1]]

    # actual tunable parameters
    tune_params["block_size_x"] = [1, 2, 4, 8, 16] + [32 * i for i in range(1, 33)]
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [i for i in range(1, 11)]
    tune_params["tile_size_y"] = [i for i in range(1, 11)]

    tune_params["temporal_tiling_factor"] = [i for i in range(1, 11)]

    max_tfactor = max(tune_params["temporal_tiling_factor"])
    tune_params["max_tfactor"] = [max_tfactor]
    tune_params["loop_unroll_factor_t"] = [i for i in range(1, max_tfactor + 1)]

    tune_params["sh_power"] = [0, 1]
    # tune_params["blocks_per_sm"] = [0,1,2,3,4]

    return tune_params, max_tfactor


def get_input_data(problem_size, max_tfactor):
    input_width = problem_size[0] + 2 * max_tfactor
    input_height = problem_size[1] + 2 * max_tfactor

    # setup main input/output data with a zero border around the input
    temp_src = np.zeros((input_height, input_width), dtype=np.float32)
    temp_src[max_tfactor:-max_tfactor, max_tfactor:-max_tfactor] = np.random.random(problem_size) + 324
    power = np.zeros((input_height, input_width), dtype=np.float32)
    power[max_tfactor:-max_tfactor, max_tfactor:-max_tfactor] = np.random.random(problem_size)
    temp_dst = np.zeros(problem_size, dtype=np.float32)

    return temp_src, power, temp_dst


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
    problem_size = (4096, 4096)

    tune_params, max_tfactor = get_tunable_parameters(problem_size)

    temp_src, power, temp_dst = get_input_data(problem_size, max_tfactor)

    # setup arguments
    step_div_cap, Rx_1, Ry_1, Rz_1 = get_input_arguments(*problem_size)
    args = [power, temp_src, temp_dst, Rx_1, Ry_1, Rz_1, step_div_cap]

    if lang == "CUDA":
        kernel_file = "/hotspot_milo.cu"
        max_shared_memory_per_block = 49152
    else:
        kernel_file = "/hotspot_milo.cu.hip"
        max_shared_memory_per_block = 65536

    with open(os.path.dirname(os.path.realpath(__file__)) + kernel_file, "r") as f:
        kernel_string = f.read()

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]
    restrictions = [
        "block_size_x*block_size_y >= 32",
        "temporal_tiling_factor % loop_unroll_factor_t == 0",
        f"block_size_x*block_size_y <= {max_threads}",
        # f"blocks_per_sm == 0 or (((block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4) * blocks_per_sm <= {dev['max_shared_memory']})",
        f"(block_size_x*tile_size_x + temporal_tiling_factor * 2) * (block_size_y*tile_size_y + temporal_tiling_factor * 2) * (2+sh_power) * 4 <= {max_shared_memory_per_block}",
    ]

    """class RegisterObserver(BenchmarkObserver):
        def get_results(self):
            return {"num_regs": self.dev.current_module.get_function("calculate_temp").num_regs}

    observer = [RegisterObserver()]"""
    observer = None

    # setup metric
    metrics = OrderedDict()

    # Temporal tiling introduces redundant work but saves data transfers
    # needed when doing the same work in multiple kernels
    # in the GFLOP/s calculation we don't count this double work, but only the 'real' work counts
    metrics["GFLOP/s"] = (
        lambda p: (p["temporal_tiling_factor"] * 15 * problem_size[0] * problem_size[1]) / 1e9 / (p["time"] / 1e3)
    )
    # metrics["reg"] = lambda p : p["num_regs"]

    base_cachepath = f"../cachefiles/hotspot_milo/{device_name.upper()}"

    # start tuning
    start = time.time()
    results, env = kernel_tuner.tune_kernel(
        "calculate_temp",
        kernel_string,
        problem_size,
        args,
        tune_params,
        iterations=32,
        metrics=metrics,
        grid_div_y=grid_div_y,
        grid_div_x=grid_div_x,
        restrictions=restrictions,
        cache=base_cachepath,
        observers=observer,
        lang=lang,
        device=0,
        verbose=verbose,
        quiet=quiet,
        strategy=strategy,
        strategy_options=strategy_options,
        simulation_mode=simulation_mode,
        objective="GFLOP/s",
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
        raise ValueError(f"Usage: python hotspot_milo.py [language ('HIP' or 'CUDA')] [device name], given: {sys.argv}")

    if language not in ("HIP", "CUDA"):
        raise ValueError(f"{language} not valid, specify HIP or CUDA")

    tune(device_name=device_name, lang=language)
