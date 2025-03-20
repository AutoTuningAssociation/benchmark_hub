import json


def adjust_scores(input_file, output_file):
    # Read JSON file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Extract all score values
    scores = [entry["measurements"][0]["value"] for entry in data["results"]]

    # # Find the lowest score
    # min_score = min(scores) - 0.001  # to avoid division by zero

    # # Offset each score by the lowest score
    # for entry in data["results"]:
    #     entry["measurements"][0]["value"] -= min_score

    # Inverse each score
    for entry in data["results"]:
        entry["measurements"][0]["value"] = entry["measurements"][0]["value"] * -1

    # Write modified data to output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


# Example usage
base_path = "/Users/fjwillemsen/Downloads/new_0.95_10x50x/"
input_files = [
    "hyperparamtuning_paper_bruteforce_diff_evo_T4.json",
    "hyperparamtuning_paper_bruteforce_dual_annealing_T4.json",
    "hyperparamtuning_paper_bruteforce_genetic_algorithm_T4.json",
    "hyperparamtuning_paper_bruteforce_pso_T4.json",
    "hyperparamtuning_paper_bruteforce_simulated_annealing_T4.json",
]
for i in input_files:
    adjust_scores(base_path + i, base_path + i.replace(".json", "_C.json"))
