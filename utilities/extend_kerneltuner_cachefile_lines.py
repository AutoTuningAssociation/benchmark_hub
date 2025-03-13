"""Utility script to extend a Kernel Tuner cachefile by using another cachefile that has those lines. Use with caution."""

import json
import numbers
from copy import deepcopy
from pathlib import Path
from warnings import warn

# set the files to use
basepath = Path(__file__).parent.parent / "cachefiles"
target_infile = basepath / "convolution_milo" / "A6000_extended.json"
target_outfile = basepath / "convolution_milo" / "A6000_extended_2.json"
extra_sourcefile = basepath / "convolution_milo" / "A4000.json"

# load the JSON files
with target_infile.open() as fp:
    target: dict = json.load(fp)
    new_target = deepcopy(target)
with extra_sourcefile.open() as fp:
    extra_source: dict = json.load(fp)["cache"]

# find the cachelines missing from the target
missing_cachelines = {}
for config_string, config in extra_source.items():
    if config_string not in target["cache"]:
        missing_cachelines[config_string] = config

# all lines in the target must be present in the source for relative difference lookup
assert all(config_string in extra_source for config_string in target["cache"].keys())

# add the missing cachelines to the new target using the relative difference between similar lines in source and target
print(f"Adding {len(missing_cachelines)} missing cachelines")
for missing_config_string, missing_config in missing_cachelines.items():
    # find the closest configuration in the target
    closest_config_string = min(
        target["cache"].keys(),
        key=lambda config_string: sum(
            abs(int(a) - int(b)) for a, b in zip(missing_config_string.split(","), config_string.split(","))
        ),
    )
    closest_config = target["cache"][closest_config_string]

    # find the corresponding configuration in the extra source
    source_closest_config = extra_source[closest_config_string]

    # create a new configuration by applying the relative difference
    new_config = deepcopy(missing_config)

    # change the values for target based on the relative difference between target, source base and source extra
    def change_relatively(target_base, source_base, source_extra):
        """Get a new value for target based on the relative difference."""
        # check if there are any error values
        if isinstance(target_base, str):
            return target_base
        elif isinstance(source_extra, str):
            return source_extra
        elif isinstance(source_base, str):
            return source_base
        # make sure all are the same type
        if isinstance(target_base, int) and isinstance(source_base, float) and isinstance(source_extra, float):
            target_base = float(target_base)
        assert (
            type(target_base) == type(source_base) == type(source_extra)
        ), f"{type(target_base)} ({target_base}) != {type(source_base)} ({source_base}) != {type(source_extra)} ({source_extra})"
        if isinstance(target_base, (list, tuple)):
            # if we're dealing with lists, go recursive
            assert len(target_base) == len(source_base) == len(source_extra)
            return [change_relatively(target_base[i], source_base[i], source_extra[i]) for i in range(len(target_base))]
        # final check for the type
        if not isinstance(target_base, numbers.Real):
            raise ValueError(
                f"Relative value change is not possible for non-numeric values of type {type(target_base)} ({target_base})"
            )
        # since we're dealing with numbers, we can do the relative value change
        try:
            fraction = source_extra / source_base
            return target_base * fraction
        except ZeroDivisionError:
            return target_base

    # mandatory keys
    for key in ["time"]:
        new_config[key] = change_relatively(closest_config[key], source_closest_config[key], missing_config[key])

    # optional keys
    for key in [
        "times",
        "compile_time",
        "verification_time",
        "benchmark_time",
        "strategy_time",
        "framework_time",
        "GFLOP/s",
    ]:
        if key in closest_config and key in source_closest_config and key in missing_config:
            new_config[key] = change_relatively(closest_config[key], source_closest_config[key], missing_config[key])
        else:
            warn(f"Key {key} missing, not adjusted relatively but kept as in missing_config", RuntimeWarning)

    # if time was an error, set times to empty
    if isinstance(new_config["time"], str):
        new_config["times"] = []

    # add the new configuration to the new target
    new_target["cache"][missing_config_string] = new_config

# check that the extension is succesful
assert len(new_target["cache"]) == len(
    extra_source
), f"Lengths don't match; target: {len(new_target['cache'])}, source: {len(extra_source)} (can also happen due to differences in restrictions`)"

# write to the target file
with target_outfile.open("w+") as fp:
    json.dump(new_target, fp)
