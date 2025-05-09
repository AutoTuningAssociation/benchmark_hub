"""Script to adjust times to remove inner array starting with zero"""

import json


def adjust_scores(input_file, output_file):
    # Read JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Iterate over cache and invert scores
    for key, entry in data["cache"].items():
        if "times" not in entry or isinstance(entry["time"], str):
            continue
        # remove inner array starting with zero
        entry["times"] = [t[1] for t in entry["times"]]
        # recalculate average time
        entry["time"] = sum(entry["times"]) / len(entry["times"])

    # Write modified data to output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


# Example usage
base_path = "../cachefiles/"
input_files = [
    "convolution_milo/W7800.json",
    "dedispersion_milo/W7800.json",
    "hotspot_milo/W7800.json",
    "gemm_milo/W7800.json",
]
for i in input_files:
    print(f"Processing {i}")
    adjust_scores(base_path + i, base_path + i.replace(".json", "_C.json"))
