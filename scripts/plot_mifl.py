import json
import matplotlib.pyplot as plt
import csv
import sys
import os
import math
import numpy as np


def read_csv(file_path):
    data = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            data.append(
                [int(row[0])] + [math.inf if x == "nan" else eval(x) for x in row[1:]]
            )
    return data


def count_mifl(row, critical_value):
    critical_value = 0
    mi = row[1]
    lower_bound = np.percentile(mi, critical_value * 100)
    upper_bound = np.percentile(mi, (1 - critical_value) * 100)
    count = len([x for x in mi if lower_bound <= x <= upper_bound])
    print(mi, critical_value, lower_bound, upper_bound)
    print(count)
    return count


def plot_metrics(run_id):
    config_path = os.path.join(run_id, "config.json")
    plots_dir = os.path.join(os.path.dirname(config_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    config_data = json.load(open(config_path))
    critical_value = config_data["critical_value"]
    if config_data["mi_type"] == "mi_gauss":
        distributed_mi_gauss_file = os.path.join(
            os.path.dirname(config_path), "metrics_distributed_fit_mi_gauss.csv"
        )
        distributed_mi_gauss_data = read_csv(distributed_mi_gauss_file)
        distributed_mi_gauss_data = [
            count_mifl(row, critical_value) for row in distributed_mi_gauss_data
        ]

        plt.figure(figsize=(12, 6))
        plt.plot(distributed_mi_gauss_data)
        plt.title(f"MIFL Gauss {critical_value} (Run ID: {run_id})")
        plt.xlabel("Round")
        plt.ylabel("Number of Models")
        plt.savefig(
            os.path.join(plots_dir, "mifl_mi_gauss.png"),
            bbox_inches="tight",
        )
        plt.close()

    elif config_data["mi_type"] == "mi_cat":
        distributed_mi_cat_file = os.path.join(
            os.path.dirname(config_path), "metrics_distributed_fit_mi_cat.csv"
        )
        distributed_mi_cat_data = read_csv(distributed_mi_cat_file)
        distributed_mi_cat_data = [
            count_mifl(row, critical_value) for row in distributed_mi_cat_data
        ]
        plt.figure(figsize=(12, 6))
        plt.plot(distributed_mi_cat_data)
        plt.title(f"MIFL Cat {critical_value} (Run ID: {run_id})")
        plt.xlabel("Round")
        plt.ylabel("Number of Models")
        plt.savefig(
            os.path.join(plots_dir, "mifl_mi_cat.png"),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    plot_metrics(run_id)
