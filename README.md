# FAIR Benchmark Hub for Auto-Tuning
This repository is a FAIR-compliant hub for GPU auto-tuning benchmarking resources. 
It provides fully brute-forced search spaces (in original and T4 format) and their source files (kernel files and T1 input files). 
Anyone can contribute by providing new kernels and brute-forced search spaces on new configurations, please do so!

The brute-forced files can be used with the [autotuning-methodology](https://github.com/AutoTuningAssociation/autotuning_methodology) to compare optimization algorithms accross a wide variety of search spaces, with T1/T4 compliant auto-tuning frameworks, and in addition with [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner) for hyperparameter tuning optimization algorithms, without constant access to the original hardware.
Please see ["FAIR Sharing of Data in Autotuning Research (Vision Paper)"](https://doi.org/10.1145/3629527.3651429) for more information on the T1 and T4 formats.

## Automatic compression and decompression
To automatically compress new cachefiles when committing and decompress new cachefiles when checking out, run `git config --local core.hooksPath .githooks/`. 
To get the decompressed files after cloning, run the above command and checkout with `git checkout main`. 

## Search spaces overview
24 fully brute-forced search spaces are currently available, as a product of the following kernels and GPUs. 
For more information on the kernels and GPUs, see [below](#Additional-information).

Kernels:
- GEMM
- Convolution
- Hotspot
- Dedispersion

GPUs:
- Nvidia A100
- Nvidia A4000
- NVIDIA A6000
- AMD MI250X
- AMD W6600
- AMD W7800

**Important:** this is a live repository, subject to change. For persistent identifiers use the DOI of a specific release. If you have questions or suggestions please submit an issue.

## File structure
- `kernels` contains for each kernel the contains the kernel files, T1 input format JSON file, and the script for the auto-tuning. 
- `cachefiles` contains the brute-forced search spaces. Each kernel has its own folder, which contains both a T4 format output file and an original cache file for each GPU the kernel has been brute-forced on. 
- `utilities` contains utility scripts.

## Additional information 
Additional information on the benchmarks is provided here.

### Kernels
The four kernels are the dedispersion, convolution, hotspot, and GEMM kernels as used in ["Bringing Auto-Tuning to HIP: Analysis of Tuning Impact and Difficulty on AMD and Nvidia GPUs"](https://doi.org/10.5281/zenodo.11617999), widely used in astronomy, image processing, material science, and linear algebra respectively.
A brief description of each is provided:
- Dedispersion is a signal processing kernel that reconstructs radio signals distorted by interstellar dispersion by applying a range of dispersion measures to time-domain samples across multiple frequency channels. 
- The 2D Convolution kernel performs image filtering by computing weighted sums over image regions. 
- Hotspot is a thermal simulation kernel that estimates processor temperature by iteratively solving differential equations based on simulated power and initial temperature inputs, producing a temperature grid as output. 
- GEMM (General Matrix-Matrix Multiplication) is a widely-used linear algebra operation implemented in CLBlast for large dense matrices. 

### GPUs and hardware
The hardware used are the following:
- Nvidia A100 of [DAS6](https://www.cs.vu.nl/das/)
- Nvidia A4000 of [DAS6](https://www.cs.vu.nl/das/)
- NVIDIA A6000 of [DAS6](https://www.cs.vu.nl/das/)
- AMD MI250X of [LUMI](https://www.lumi-supercomputer.eu)
- AMD W6600 of [DAS6](https://www.cs.vu.nl/das/)
- AMD W7800 of [DAS6](https://www.cs.vu.nl/das/)
