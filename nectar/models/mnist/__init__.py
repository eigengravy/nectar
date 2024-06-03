from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset


def apply_transforms(batch):
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


def get_dataset(num_splits: int):
    federated_dataset = FederatedDataset(
        dataset="mnist", partitioners={"train": num_splits}
    )
    centralized_testset = federated_dataset.load_split("test")
