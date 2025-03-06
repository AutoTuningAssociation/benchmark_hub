# Benchmark Hub for Auto-Tuning
This repository is a hub for GPU auto-tuning benchmarking resources. 
It provides fully brute-forced search spaces and their source files (kernel files and T1 input files). 
Anyone can contribute by providing new kernels and brute-forced search spaces on new configurations, please do so!

The brute-forced files can be used with the [autotuning-methodology](https://github.com/AutoTuningAssociation/autotuning_methodology) to compare optimization algorithms accross a wide variety of search spaces, and with [Kernel Tuner]() for hyperparameter tuning optimization algorithms, without constant access to the original hardware.

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

This means that 16 fully brute-forced search spaces are currently available. 
