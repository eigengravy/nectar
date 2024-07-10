from datetime import datetime
import os
import pickle
from typing import Dict
import uuid
from flwr.common.typing import Scalar
import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from datasets.utils.logging import disable_progress_bar

from nectar.client import FlowerClient
from nectar.utils.metrics import (
    evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn,
)
from nectar.utils.params import set_params

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": cfg.fit_config.epochs,
            "mi_type": cfg.fit_config.mi_type,
        }
        return config

    def get_client_fn(get_client_loader):
        def client_fn(cid: int) -> fl.client.Client:
            trainloader, valloader = get_client_loader(cid)
            model = instantiate(cfg.model).to(DEVICE)
            optimizer = instantiate(cfg.fit_config.optimizer, model.parameters())
            return FlowerClient(
                cid,
                trainloader,
                valloader,
                model,
                optimizer,
                criterion=instantiate(cfg.fit_config.criterion),
                distiller=instantiate(cfg.fit_config.distiller),
                device=DEVICE,
                train_fn=cfg.dataset.train_fn,
                mi_fn=cfg.dataset.mi_fn,
                test_fn=cfg.dataset.test_fn,
            ).to_client()

        return client_fn

    def get_evaluate_fn(testloader):
        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
        ):
            model = instantiate(cfg.model).to(DEVICE)
            set_params(model, parameters)
            loss, accuracy = instantiate(
                cfg.dataset.test_fn, model, testloader, device=DEVICE
            )
            return loss, {"accuracy": accuracy}

        return evaluate

    testloader, get_client_loader = instantiate(
        cfg.dataset.load_dataset,
        dict(instantiate(cfg.partitioners)),
        batch_size=cfg.fit_config.batch_size,
        test_size=0.1,
    )

    strategy = instantiate(
        cfg.strategy,
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testloader),
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(get_client_loader),
        num_clients=cfg.num_clients,
        client_resources=cfg.client_resources,
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds, round_timeout=cfg.round_timeout
        ),
        strategy=strategy,
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
        ray_init_args={"include_dashboard": True},
    )

    save_path = HydraConfig.get().runtime.output_dir
    pickle.dump(
        history,
        open(f"{save_path}/history.pkl", "wb"),
    )


if __name__ == "__main__":
    main()
