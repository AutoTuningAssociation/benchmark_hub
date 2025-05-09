from pathlib import Path

from kernel_tuner.cache.cli_tools import convert_t4

basepath = Path(__file__).parent.parent / "cachefiles"
directories = ["convolution_milo", "dedispersion_milo", "gemm_milo", "hotspot_milo"]

# basepath = Path("/Users/fjwillemsen/Downloads")
# directories = ["new_0.95_10x50x"]

for directory in directories:
    print(f"Converting files in {directory}")
    dirpath = Path(basepath / directory)
    assert dirpath.is_dir(), f"Not a directory: {dirpath}"
    for infile in dirpath.iterdir():
        if infile.suffix.endswith("json") and not (infile.stem.endswith("_T4") or infile.stem.endswith("_C")):
            outfile = infile.with_stem(infile.stem + "_T4")
            if outfile.exists():
                print(f"  | skipping {infile.stem}, already exists")
            else:
                print(f"  | converting {infile.stem}")
                convert_t4(infile, outfile)
