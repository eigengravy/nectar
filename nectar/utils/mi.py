from enum import Enum
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
import math


def mutual_information(x: Tensor, y: Tensor, mi_type: str) -> Tensor:
    """
    Calculate the mutual information between two distributions.

    Args:
    x (torch.Tensor): First tensor
    y (torch.Tensor): Second tensor
    mi_type (str): Type of the distribution

    Returns:
    torch.Tensor: Mutual information
    """
    if mi_type == "CATEGORICAL":
        return categorical_mutual_information(x, y)
    elif mi_type == "GAUSSIAN":
        return gaussian_mutual_information(x, y)
    else:
        raise ValueError(
            "Invalid distribution type. Must be 'CATEGORICAL' or 'GAUSSIAN'."
        )


def categorical_mutual_information(x: Tensor, y: Tensor) -> Tensor:
    """
    Calculate the mutual information between two Categorical distributions.

    Args:
    x (torch.Tensor): First tensor of shape (batch_size, num_categories)
    y (torch.Tensor): Second tensor of shape (batch_size, num_categories)

    Returns:
    torch.Tensor: Mutual information
    """

    def categorical_entropy(p: Tensor) -> Tensor:
        return -torch.sum(
            F.softmax(p, dim=-1) * torch.log(F.softmax(p, dim=-1) + 1e-8), dim=-1
        )

    h_x = categorical_entropy(x.mean(dim=0))
    h_y = categorical_entropy(y.mean(dim=0))
    h_xy = categorical_entropy(
        torch.bmm(x.unsqueeze(2), y.unsqueeze(1)).mean(dim=0).view(-1)
    )

    return h_x + h_y - h_xy


def gaussian_mutual_information(x: Tensor, y: Tensor) -> Tensor:
    """
    Calculate the mutual information between two Gaussian distributions with covariance.
    Args:
    x (torch.Tensor): First tensor of shape (batch_size, feature_dim)
    y (torch.Tensor): Second tensor of shape (batch_size, feature_dim)

    Returns:
    torch.Tensor: Mutual information
    """

    sigma_g = torch.std(x, dim=0, keepdim=True)  # Shape: (1, n_params)
    sigma_k = torch.std(y, dim=0, keepdim=True)  # Shape: (1, n_params)

    E_Fg = torch.mean(x, dim=0, keepdim=True)
    E_Fk = torch.mean(y, dim=0, keepdim=True)
    # Calculate rho_gk over the batch dimension
    rho_gk = torch.mean((x - E_Fg) * (y - E_Fk), dim=0) / (
        sigma_g * sigma_k + 1e-8
    )  # Shape: (n_params,)

    return -0.5 * torch.mean(torch.log(1 - rho_gk**2))


# def gaussian_mutual_information(x: Tensor, y: Tensor) -> Tensor:
#     """
#     Calculate the mutual information between two Gaussian distributions with covariance.
#     Args:
#     x (torch.Tensor): First tensor of shape (batch_size, feature_dim)
#     y (torch.Tensor): Second tensor of shape (batch_size, feature_dim)

#     Returns:
#     torch.Tensor: Mutual information
#     """

#     def gaussian_entropy(cov: Tensor) -> Tensor:
#         return 0.5 * torch.logdet(2 * math.pi * math.e * cov)

#     h_x = gaussian_entropy(torch.cov(x.T))
#     h_y = gaussian_entropy(torch.cov(y.T))
#     h_xy = gaussian_entropy(torch.cov(torch.cat([x, y], dim=1).T))

#     return h_x + h_y - h_xy


# def gaussian_entropy_with_covariance(mu, cov, eps=1e-8):
#     """
#     Calculate the entropy of a multivariate Gaussian distribution.

#     Args:
#     mu (torch.Tensor): Mean vector of shape (feature_dim,)
#     cov (torch.Tensor): Covariance matrix of shape (feature_dim, feature_dim)
#     eps (float): Small value to add to diagonal for numerical stability

#     Returns:
#     torch.Tensor: Entropy of the multivariate Gaussian distribution
#     """
#     feature_dim = mu.shape[0]
#     cov_reg = cov + torch.eye(feature_dim, device=cov.device) * eps
#     eigvals = torch.linalg.eigvalsh(cov_reg)
#     logdet = torch.sum(torch.log(eigvals))
#     return 0.5 * (feature_dim * (1.0 + math.log(2 * math.pi)) + logdet)


def _gen_random_tensors(batch_size: int, num_categories: int) -> Tuple[Tensor, Tensor]:
    x = torch.randn(batch_size, num_categories)
    y = torch.randn(batch_size, num_categories)
    return x, y


def _gen_correlated_tensors(
    batch_size: int, num_categories: int
) -> Tuple[Tensor, Tensor]:
    mean_x = torch.zeros(num_categories)
    temp_matrix = torch.randn(num_categories, num_categories)
    cov_x = torch.mm(temp_matrix, temp_matrix.t()) + torch.eye(num_categories)
    cov_x /= cov_x.max()
    x = torch.distributions.MultivariateNormal(mean_x, cov_x).sample((batch_size,))
    y = torch.zeros(batch_size, num_categories)
    for i in range(num_categories):
        weights = torch.randn(num_categories)
        weights /= weights.sum()
        y[:, i] = torch.mm(x, weights.unsqueeze(1)).squeeze() + 0.1 * torch.randn(
            batch_size
        )
    y = (y - y.mean(dim=0)) / y.std(dim=0)
    return x, y


if __name__ == "__main__":
    batch_size = 1000
    num_categories = 200

    print("Random Tensors")
    x, y = _gen_random_tensors(batch_size, num_categories)

    # cat_mi = categorical_mutual_information(x, y)
    # print(f"Categorical Mutual Information: {cat_mi.item()}")
    # gauss_mi = gaussian_mutual_information(x, y)
    # print(f"Gaussian Mutual Information: {gauss_mi.item()}")
    # temp = gaussian_mutual_information_with_covariance(x,y)
    # print(f"Gaussian Mutual Information: {temp.item()}")
    print(mutual_information(x, y, "CATEGORICAL").item())
    print(mutual_information(x, y, "GAUSSIAN").item())

    print("-" * 50)

    print("Correlated Tensors")
    x, y = _gen_correlated_tensors(batch_size, num_categories)

    # cat_mi = categorical_mutual_information(x, y)
    # print(f"Categorical Mutual Information: {cat_mi.item()}")
    # gauss_mi = gaussian_mutual_information(x, y)
    # print(f"Gaussian Mutual Information: {gauss_mi.item()}")
    # temp = gaussian_mutual_information_with_covariance(x,y)
    # print(f"Gaussian Mutual Information: {temp.item()}")
    print(mutual_information(x, y, "CATEGORICAL").item())
    print(mutual_information(x, y, "GAUSSIAN").item())

    print("-" * 50)

    print("Same Tensors")
    y = x

    # cat_mi = categorical_mutual_information(x, y)
    # print(f"Categorical Mutual Information: {cat_mi.item()}")
    # gauss_mi = gaussian_mutual_information(x, y)
    # print(f"Gaussian Mutual Information: {gauss_mi.item()}")
    # temp = gaussian_mutual_information_with_covariance(x,y)
    # print(f"Gaussian Mutual Information: {temp.item()}")
    print(mutual_information(x, y, "CATEGORICAL").item())
    print(mutual_information(x, y, "GAUSSIAN").item())
