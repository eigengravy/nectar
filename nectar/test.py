from omegaconf import DictConfig, OmegaConf
import hydra

from hydra.utils import instantiate
from flwr_datasets import FederatedDataset


@hydra.main(version_base=None, config_path="../config", config_name="base")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))

    dataset = FederatedDataset(
        dataset=cfg.dataset.name,
        partitioners=dict(instantiate(cfg.partitioners)),
    )

    print(dataset)
    print(dataset.load_partition(1))


if __name__ == "__main__":
    my_app()
