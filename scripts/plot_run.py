import matplotlib.pyplot as plt
import csv
import sys
import os
import math


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


def plot_metric(data, title, ylabel, file_name):
    # print(data)
    rounds = [row[0] for row in data]
    metric_values = [row[1:] for row in data]
    plt.figure(figsize=(12, 6))
    for i in range(len(metric_values[0])):
        client_metric = [metric[i] for metric in metric_values]
        plt.plot(rounds, client_metric, label=f"Client {i+1}")
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def plot_metrics(run_id):
    config_path = os.path.join(run_id, "config.json")
    plots_dir = os.path.join(os.path.dirname(config_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    centralized_accuracy_file = os.path.join(
        os.path.dirname(config_path), "metrics_centralized_accuracy.csv"
    )
    centralized_accuracy_data = read_csv(centralized_accuracy_file)
    centralized_rounds = [row[0] for row in centralized_accuracy_data]
    centralized_accuracy = [row[1] for row in centralized_accuracy_data]

    plt.figure(figsize=(12, 6))
    plt.plot(centralized_rounds, centralized_accuracy, label="Centralized Accuracy")
    plt.title(f"Centralized Accuracy (Run ID: {run_id})")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "metrics_centralized_accuracy.png"), bbox_inches="tight"
    )
    plt.close()

    distributed_accuracy_file = os.path.join(
        os.path.dirname(config_path), "metrics_distributed_accuracy.csv"
    )
    distributed_accuracy_data = read_csv(distributed_accuracy_file)
    distributed_rounds = [row[0] for row in distributed_accuracy_data]
    distributed_accuracy = [row[1] for row in distributed_accuracy_data]

    plt.figure(figsize=(12, 6))
    plt.plot(distributed_rounds, distributed_accuracy, label="Distributed Accuracy")
    plt.title(f"Distributed Accuracy (Run ID: {run_id})")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "metrics_distributed_accuracy.png"), bbox_inches="tight"
    )
    plt.close()

    centralized_loss_file = os.path.join(
        os.path.dirname(config_path), "losses_centralized.csv"
    )
    centralized_loss_data = read_csv(centralized_loss_file)
    centralized_rounds = [row[0] for row in centralized_loss_data]
    centralized_loss = [row[1] for row in centralized_loss_data]

    plt.figure(figsize=(12, 6))
    plt.plot(centralized_rounds, centralized_loss, label="Centralized Loss")
    plt.title(f"Centralized Loss (Run ID: {run_id})")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "metrics_centralized_loss.png"), bbox_inches="tight"
    )
    plt.close()

    distributed_loss_file = os.path.join(
        os.path.dirname(config_path), "losses_distributed.csv"
    )
    distributed_loss_data = read_csv(distributed_loss_file)
    distributed_rounds = [row[0] for row in distributed_loss_data]
    distributed_loss = [row[1] for row in distributed_loss_data]

    plt.figure(figsize=(12, 6))
    plt.plot(distributed_rounds, distributed_loss, label="Distributed Loss")
    plt.title(f"Distributed Loss (Run ID: {run_id})")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(plots_dir, "metrics_distributed_loss.png"), bbox_inches="tight"
    )
    plt.close()

    # ---------------------- Plotting metrics for each client ----------------------

    distributed_loss_file = os.path.join(
        os.path.dirname(config_path), "metrics_distributed_fit_loss.csv"
    )
    distributed_loss_data = read_csv(distributed_loss_file)
    plot_metric(
        distributed_loss_data,
        f"Distributed Loss (Run ID: {run_id})",
        "Loss",
        os.path.join(plots_dir, "metrics_distributed_fit_loss.png"),
    )

    distributed_accuracy_file = os.path.join(
        os.path.dirname(config_path), "metrics_distributed_fit_accuracy.csv"
    )
    distributed_accuracy_data = read_csv(distributed_accuracy_file)
    plot_metric(
        distributed_accuracy_data,
        f"Distributed Accuracy (Run ID: {run_id})",
        "Accuracy",
        os.path.join(plots_dir, "metrics_distributed_fit_accuracy.png"),
    )

    distributed_mi_gauss_file = os.path.join(
        os.path.dirname(config_path), "metrics_distributed_fit_mi_gauss.csv"
    )
    distributed_mi_gauss_data = read_csv(distributed_mi_gauss_file)
    plot_metric(
        distributed_mi_gauss_data,
        f"Distributed MI Gauss (Run ID: {run_id})",
        "MI Gauss",
        os.path.join(plots_dir, "metrics_distributed_fit__mi_gauss.png"),
    )

    distributed_mi_cat_file = os.path.join(
        os.path.dirname(config_path), "metrics_distributed_fit_mi_cat.csv"
    )
    distributed_mi_cat_data = read_csv(distributed_mi_cat_file)
    plot_metric(
        distributed_mi_cat_data,
        f"Distributed MI Cat (Run ID: {run_id})",
        "MI Cat",
        os.path.join(plots_dir, "metrics_distributed_fit_mi_cat.png"),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <run_id>")
        sys.exit(1)

    run_id = sys.argv[1]
    plot_metrics(run_id)
