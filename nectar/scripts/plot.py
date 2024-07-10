import os
import pickle
import sys
import matplotlib.pyplot as plt
import csv

import numpy as np
from omegaconf import OmegaConf


def process_optimifl(history, bitmap, clients):

    return bitmap, clients


def process_other_strategy(config, history, bitmap, clients):
    critical_value = config["strategy"]["critical_value"]

    for round_num, (mi_values, client_ids) in enumerate(
        zip(
            history.metrics_distributed_fit["client_mi"],
            history.metrics_distributed_fit["client_cid"],
        )
    ):
        mi = np.array(list(map(float, mi_values.split(","))))
        lower_bound, upper_bound = np.percentile(
            mi, [critical_value * 100, (1 - critical_value) * 100]
        )

        mask = (lower_bound <= mi) & (mi <= upper_bound)
        clients.append(np.sum(mask))
        bitmap[round_num, mask] = True


for path in sys.argv[1:]:
    if not os.path.isdir(path):
        continue
    try:
        config = OmegaConf.load(os.path.join(path, ".hydra/config.yaml"))
        history = pickle.load(open(os.path.join(path, "history.pkl"), "rb"))

        strategy = config["strategy"]["_target_"]
        losses = list(map(lambda x: x[1], history.losses_centralized))
        accuracies = list(map(lambda x: x[1], history.metrics_centralized["accuracy"]))

        num_clients = config["num_clients"]
        num_rounds = config["num_rounds"]
        bitmap = np.zeros((num_rounds, num_clients), dtype=bool)
        client_mi = np.full((num_rounds, num_clients), np.nan)

        clients = [
            len(ids.split(","))
            for (_, ids) in history.metrics_distributed_fit["client_cid"]
        ]

        for round_num, (client_ids, client_mis) in enumerate(
            zip(
                history.metrics_distributed_fit["client_cid"],
                history.metrics_distributed_fit["client_mi"],
            )
        ):
            for _client, _mi in zip(
                map(int, client_ids[1].split(",")), map(float, client_mis[1].split(","))
            ):
                bitmap[round_num, _client] = True
                client_mi[round_num, _client] = _mi

        count = np.sum(bitmap, axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(count)), count)
        plt.title("Count")
        plt.xlabel("Client")
        plt.ylabel("#")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(path, "count.png"), bbox_inches="tight")
        plt.close()

        mi = list(map(lambda x: x[1], history.metrics_distributed_fit["mi"]))

        plt.figure(figsize=(12, 6))
        plt.plot(losses)
        plt.title("Centralized Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(path, "losses_centralized.png"), bbox_inches="tight")
        plt.close()

        with open(os.path.join(path, "losses_centralized.csv"), "w") as f:
            w = csv.writer(f)
            w.writerows([[loss] for loss in losses])

        plt.figure(figsize=(12, 6))
        plt.plot(accuracies)
        plt.title("Centralized Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(path, "accuracy_centralized.png"), bbox_inches="tight")
        plt.close()

        with open(os.path.join(path, "accuracy_centralized.csv"), "w") as f:
            w = csv.writer(f)
            w.writerows([[acc] for acc in accuracies])

        plt.figure(figsize=(12, 6))
        plt.plot(clients)
        plt.title("Number of Clients")
        plt.xlabel("Round")
        plt.ylabel("Clients")
        plt.legend()
        plt.savefig(os.path.join(path, "clients.png"), bbox_inches="tight")
        plt.close()

        with open(os.path.join(path, "clients.csv"), "w") as f:
            w = csv.writer(f)
            w.writerows([[client] for client in clients])

        plt.figure(figsize=(12, 6))
        plt.plot(mi)
        plt.title("Mutual Information")
        plt.xlabel("Round")
        plt.ylabel("MI")
        plt.legend()
        plt.savefig(os.path.join(path, "mi.png"), bbox_inches="tight")
        plt.close()

        with open(os.path.join(path, "mi.csv"), "w") as f:
            w = csv.writer(f)
            w.writerows([[mi] for mi in mi])

        plt.figure(figsize=(12, 6))
        plt.imshow(bitmap, cmap="gray", interpolation="nearest")
        plt.title("Bitmap")
        plt.savefig(
            os.path.join(path, "bitmap.png"),
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(client_mi)
        plt.title("Client MI")
        plt.xlabel("Round")
        plt.ylabel("MI")
        plt.legend()
        plt.savefig(os.path.join(path, "client_mi.png"), bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Error processing {path}: {e}")
        continue

# # Deprecated processing code

# clients = []
# if strategy == "nectar.strategy.optimifl.OptiMIFL":
#     for round_num, client_ids in history.metrics_distributed_fit["client_cid"]:
#         clients.append(len(client_ids.split(",")))
#         for client in map(int, client_ids.split(",")):
#             bitmap[round_num - 1, client] = True
# else:
#     critical_value = config["strategy"]["critical_value"]
#     for i, (res_mi, res_cid) in enumerate(
#         zip(
#             history.metrics_distributed_fit["client_mi"],
#             history.metrics_distributed_fit["client_cid"],
#         )
#     ):
#         _mi = list(map(float, res_mi[1].split(",")))
#         lower_bound_mi, upper_bound_mi = np.percentile(
#             _mi, [critical_value * 100, (1 - critical_value) * 100]
#         )
#         clients.append(
#             len([m for m in _mi if lower_bound_mi <= m <= upper_bound_mi])
#         )
#         client_ids = list(map(int, res_cid[1].split(",")))
#         for j, mi_val in enumerate(_mi):
#             if lower_bound_mi <= mi_val <= upper_bound_mi:
#                 bitmap[i, client_ids[j]] = 1
