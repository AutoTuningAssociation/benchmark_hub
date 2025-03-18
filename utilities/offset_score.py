import json


def adjust_scores(input_file, output_file):
    # Read JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Iterate over cache and invert scores
    for key, entry in data["cache"].items():
        entry["score"] *= -1
        entry["scores"] = [-s for s in entry["scores"]]

    # Write modified data to output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


# Example usage
base_path = "/Users/fjwillemsen/Downloads/new_0.95_10x50x/"
input_files = [
    "hyperparamtuning_paper_bruteforce_dual_annealing.json",
    "hyperparamtuning_paper_bruteforce_pso.json",
    "hyperparamtuning_paper_bruteforce_simulated_annealing.json",
]
for i in input_files:
    adjust_scores(base_path + i, base_path + i.replace(".json", "_C.json"))
