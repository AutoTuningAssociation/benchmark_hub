"""Get the durations of all the runs."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tabulate import tabulate

gpus_used = ["A100", "A4000", "A6000", "MI250X", "W6600", "W7800"]


# For each cachefile in the cache directory, get the difference between the first timestamp and the last timestamp
def get_durations(cache_dir: Path) -> dict:
    """Get the durations of all the runs."""
    durations = {}
    for cachefile_dir in cache_dir.iterdir():
        if not cachefile_dir.is_dir():
            continue
        durations[cachefile_dir.stem.replace("_milo", "")] = {}
        for cachefile in cachefile_dir.glob("*.json"):
            if (
                not cachefile.is_file()
                or cachefile.stem.endswith("_T4")
                or cachefile.stem.endswith("_C")
                or cachefile.stem.endswith("_original")
            ):
                continue
            # skip the cachefile if it does not contain one of the GPU names
            if not any(gpu in cachefile.stem for gpu in gpus_used):
                continue
            print(f"Processing {cachefile}")
            with open(cachefile, "r") as f:
                data = json.load(f)
            # Get the first and last timestamp
            cache = data["cache"]
            first_key = list(cache)[0]
            first_timestamp = datetime.fromisoformat(cache[first_key]["timestamp"])
            last_key = list(cache)[-1]
            last_timestamp = datetime.fromisoformat(cache[last_key]["timestamp"])
            duration = last_timestamp - first_timestamp
            durations[cachefile_dir.stem.replace("_milo", "")][cachefile.stem] = duration
    return durations


def timedelta_dict_to_df(data):
    # Flatten the dictionary and convert timedeltas to total seconds
    records = []
    for app, devices in data.items():
        for device, delta in devices.items():
            records.append({"application": app, "device": device, "time_seconds": delta.total_seconds()})

    # Create the DataFrame
    df = pd.DataFrame(records)

    # Pivot so rows = applications, columns = devices, values = time_seconds
    return df.pivot(index="application", columns="device", values="time_seconds")


def timedelta_dict_to_latex_tabularx_hours(data):
    df = timedelta_dict_to_df(data)

    # Convert seconds to hours
    df = df / 3600
    df = df.round(1)

    # Sort columns alphabetically for consistency
    df = df[sorted(df.columns)]
    df = df.sort_index()

    # Fill missing values
    df = df.fillna("")

    # Build the LaTeX table body using tabulate
    col_names = ["Application"] + list(df.columns)
    table = [[app] + list(row) for app, row in df.iterrows()]
    latex = tabulate(table, headers=col_names, tablefmt="latex_raw")

    # Wrap with tabularx
    latex_table = (
        "\\begin{table}[htb]\n"
        "\\centering\n"
        "\\caption{Execution times in hours for each application on each GPU.}\n"
        "\\label{tab:bruteforce-times}\n"
        "\\begin{tabularx}{\\linewidth}{l" + "|X" * len(df.columns) + "}\n"
        "\\toprule\n"
        + latex.splitlines()[0]
        + " \\\\\n\\midrule\n"
        + "\n".join(line + " \\\\" for line in latex.splitlines()[2:])
        + "\n\\bottomrule\n"
        "\\end{tabularx}\n"
        "\\end{table}"
    )
    return latex_table


if __name__ == "__main__":
    # Example usage
    cache_dir = Path("../cachefiles")
    durations = get_durations(cache_dir)
    # print(durations)
    df = timedelta_dict_to_df(durations)
    print(df)
    latex_table = timedelta_dict_to_latex_tabularx_hours(durations)
    print(latex_table)
    print(f"Total duration: {round(df.to_numpy().sum() / 3600, 3)} hours")
    # for run, duration in durations.items():
    # print(f"{run}: {duration} seconds")
