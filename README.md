# Benchmark Hub for Auto-Tuning
This repository is a hub for GPU auto-tuning benchmarking resources. 
It provides fully brute-forced search spaces and their source files (kernel files and T1 input files). 
Anyone can contribute by providing new kernels and brute-forced search spaces on new configurations, please do so!

The brute-forced files can be used with the [autotuning-methodology](https://github.com/AutoTuningAssociation/autotuning_methodology) to compare optimization algorithms accross a wide variety of search spaces, and with [Kernel Tuner]() for hyperparameter tuning optimization algorithms, without constant access to the original hardware.

## Automatic compression and decompression
To automatically compress new cachefiles when committing and decompress new cachefiles when checking out, run `git config --local core.hooksPath .githooks/`. 
To get the decompressed files after cloning, run the above command and checkout with `git checkout main`.

## Search spaces overview
16 fully brute-forced search spaces are currently available, as a product of the following kernels and GPUs. 

Kernels:
- GEMM
- Convolution
- Hotspot
- Dedispersion

GPUs:
- Nvidia A100
- Nvidia A4000
- AMD MI250X
- AMD W6600

**Important:** this is a live repository, not everything is standardized. If you have questions or suggestions please submit an issue.

## File structure
- `kernels` contains for each kernel the contains the kernel files, T1 input format JSON file, and the script for the auto-tuning. 
- `cachefiles` contains the brute-forced search spaces. Each kernel has its own folder, which contains both a T4 format output file and an original cache file for each GPU the kernel has been brute-forced on. 
- `utilities` contains utility scripts. 
