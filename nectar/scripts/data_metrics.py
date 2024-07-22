from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions

fds = FederatedDataset(
    dataset="zh-plus/tiny-imagenet",
    partitioners={
        "train": DirichletPartitioner(
            num_partitions=30, alpha=0.5, partition_by="label"
        )
    },
)

partition_sizes = partition_sizes = [
    len(fds.load_partition(partition_id)) for partition_id in range(30)
]

print(partition_sizes)

partitioner = fds.partitioners["train"]

fig, ax, df = plot_label_distributions(
    partitioner,
    label_name="label",
    plot_type="bar",
    size_unit="absolute",
    partition_id_axis="x",
    legend=True,
    verbose_labels=True,
    figsize=(20, 10),
    title="Per Partition Labels Distribution",
)

fig.savefig("blah.jpg")

fig, ax, df = plot_label_distributions(
    partitioner,
    label_name="label",
    plot_type="bar",
    size_unit="percent",
    partition_id_axis="x",
    legend=True,
    verbose_labels=True,
    figsize=(20, 10),
    cmap="tab20b",
    title="Per Partition Labels Distribution",
)

fig.savefig("blah2.jpg")

fig, ax, df = plot_label_distributions(
    partitioner,
    label_name="label",
    plot_type="heatmap",
    size_unit="absolute",
    partition_id_axis="x",
    legend=True,
    verbose_labels=True,
    figsize=(20, 10),
    title="Per Partition Labels Distribution",
    plot_kwargs={"annot": True},
)

fig.savefig("blah3.jpg")
