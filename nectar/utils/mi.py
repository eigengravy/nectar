import torch


def categorical_entropy(tensor):
    probs = torch.softmax(tensor, dim=-1)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    return entropy


def gaussian_entropy(tensor):
    variance = torch.var(tensor, dim=-1) + 1e-6
    entropy = 0.5 * (1 + torch.log(2 * torch.pi * variance))
    return entropy


def mutual_information(tensor1, tensor2, dist_type="categorical"):
    if dist_type == "categorical":
        entropy_func = categorical_entropy
    elif dist_type == "gaussian":
        entropy_func = gaussian_entropy
    else:
        raise ValueError(
            "Invalid distribution type. Must be 'categorical' or 'gaussian'."
        )

    joint_tensor = torch.cat((tensor1, tensor2), dim=-1)
    joint_entropy = entropy_func(joint_tensor)
    marginal_entropy1 = entropy_func(tensor1)
    marginal_entropy2 = entropy_func(tensor2)
    mutual_info = marginal_entropy1 + marginal_entropy2 - joint_entropy

    return mutual_info


def averaged_mutual_information(tensor1, tensor2, dist_type="categorical"):
    mutual_info = mutual_information(tensor1, tensor2, dist_type)
    averaged_mi = mutual_info.mean()
    return averaged_mi


def normalized_mutual_information(tensor1, tensor2, dist_type="categorical"):
    if dist_type == "categorical":
        entropy_func = categorical_entropy
    elif dist_type == "gaussian":
        entropy_func = gaussian_entropy
    else:
        raise ValueError(
            "Invalid distribution type. Must be 'categorical' or 'gaussian'."
        )
    mutual_info = mutual_information(tensor1, tensor2, dist_type)
    entropy1 = entropy_func(tensor1)
    entropy2 = entropy_func(tensor2)
    normalized_mi = mutual_info / (entropy1 + entropy2)
    return normalized_mi


if __name__ == "__main__":
    tensor1 = torch.randn(100, 10)
    tensor2 = torch.randn(100, 10)

    # mutual_info_categorical = mutual_information(
    #     tensor1, tensor2, dist_type="categorical"
    # )
    # mutual_info_gaussian = mutual_information(tensor1, tensor2, dist_type="gaussian")

    # averaged_mi_categorical = averaged_mutual_information(
    #     tensor1, tensor2, dist_type="categorical"
    # )
    # averaged_mi_gaussian = averaged_mutual_information(
    #     tensor1, tensor2, dist_type="gaussian"
    # )

    # normalized_mi_categorical = normalized_mutual_information(
    #     tensor1, tensor2, dist_type="categorical"
    # )
    # normalized_mi_gaussian = normalized_mutual_information(
    #     tensor1, tensor2, dist_type="gaussian"
    # )

    a = normalized_mutual_information(tensor1, tensor2, dist_type="gaussian")
    b = normalized_mutual_information(tensor2, tensor1, dist_type="gaussian")

    print(a.shape)
    print(b.shape)
    print((a + b).shape)
    print(torch.cat((torch.empty(), b), dim=-1).shape)
    # print(mutual_info_categorical)
    # print(mutual_info_gaussian)
    # print(averaged_mi_categorical)
    # print(averaged_mi_gaussian)
    # print(normalized_mi_categorical)
    # print(normalized_mi_gaussian)
