from torchvision.transforms import Compose, Lambda, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


def apply_transforms(batch):
    tfs = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        ]
    )
    batch["image"] = [tfs(img) for img in batch["image"]]
    return batch


def get_dataset(num_splits: int):
    dataset = FederatedDataset(
        dataset="zh-plus/tiny-imagenet",
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=num_splits, alpha=0.5, partition_by="label"
            ),
        },
    )
    centralised_testset = dataset.load_split("valid")
    return dataset, centralised_testset
