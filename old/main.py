import hydra
import torch
import flwr as fl
from hydra.utils import instantiate
from flwr_datasets import FederatedDataset

from torch.utils.data import DataLoader
from nectar.utils.params import set_params


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg):

    federated_dataset = FederatedDataset(
        dataset=cfg.dataset.name,
        partitioners=dict(instantiate(cfg.partitioners)),
    )


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, cid, config):
        self.cid = cid
        self.trainset = trainset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = instantiate(config.model).to(self.device)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        batch, epochs = config["batch_size"], config["epochs"]
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        results = train(
            self.model, trainloader, optimizer, epochs=epochs, device=self.device
        )
        return self.get_parameters({}), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        valloader = DataLoader(self.valset, batch_size=64)
        loss, accuracy = test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    main()
