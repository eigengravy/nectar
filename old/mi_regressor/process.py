import csv
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)

from xgboost import XGBRegressor

# # Format MI metrics to CSV
# with open(sys.argv[1]) as f:
#     reader = csv.reader(f)
#     next(reader)
#     with open(sys.argv[2], "a+") as w:
#         writer = csv.writer(w)
#         writer.writerows([eval(row[1]) for row in reader])


df = pd.read_csv(sys.argv[1])
critical_value = float(sys.argv[2])


def calculate_bounds(row, critical_value):
    lower_bound = np.percentile(row, critical_value * 100)
    upper_bound = np.percentile(row, (1 - critical_value) * 100)
    return pd.Series([lower_bound, upper_bound])


df[["lower_bound", "upper_bound"]] = df.apply(
    lambda row: calculate_bounds(row, critical_value), axis=1
)

# print(df.head())

df["lower_bound_target"] = df["lower_bound"].shift(-1)
df["upper_bound_target"] = df["upper_bound"].shift(-1)

# print(df.head())

df = df.dropna()


def prepare_time_series_data(df, column_names, window_size=10):
    """
    Prepare time series data for mutual information calculation.

    Args:
    df (pandas.DataFrame): Input dataframe
    column_names (list): List of column names to use
    window_size (int): Number of time steps to include

    Returns:
    torch.Tensor: Tensor of shape (num_windows, window_size * len(column_names))
    """

    data = df[column_names].values
    num_samples = len(data) - window_size + 1

    windows = np.lib.stride_tricks.sliding_window_view(
        data, (window_size, len(column_names))
    )
    return windows.reshape(num_samples, -1)


# print("Regression on Calculated Bounds")

# X = df[["lower_bound", "upper_bound"]]
# y_lower = df["lower_bound_target"]
# y_upper = df["upper_bound_target"]

# # Split the data into training and testing sets
# X_train, X_test, y_train_lower, y_test_lower, y_train_upper, y_test_upper = (
#     train_test_split(X, y_lower, y_upper, test_size=float(sys.argv[3]), random_state=42)
# )


# print("Lower Bound")


# # print("Linear Regression")
# # Initialize and train the model for lower_bound
# model_lower = LinearRegression()
# model_lower.fit(X_train, y_train_lower)

# # Predict and evaluate the model for lower_bound
# y_pred_lower = model_lower.predict(X_test)

# print(y_test_lower.head(), pd.DataFrame(y_pred_lower).head())
# mse_lower = mean_squared_error(y_test_lower, y_pred_lower)
# rmse_lower = root_mean_squared_error(y_test_lower, y_pred_lower)
# mape_lower = mean_absolute_percentage_error(y_test_lower, y_pred_lower)
# print(f"Mean Squared Error for lower_bound: {mse_lower}")
# print(f"Root Mean Squared Error for lower_bound: {rmse_lower}")
# print(
#     f"Normalised Root Mean Squared Error for lower_bound: {rmse_lower / y_test_lower.mean()}"
# )
# print(f"Mean Absolute Percentage Error for lower_bound: {mape_lower}")

# # print("XGBoost Regression")

# # model_lower = XGBRegressor(objective="reg:squarederror")
# # model_lower.fit(X_train, y_train_lower)

# # y_pred_lower = model_lower.predict(X_test)
# # mse_lower = mean_squared_error(y_test_lower, y_pred_lower)
# # rmse_lower = root_mean_squared_error(y_test_lower, y_pred_lower)
# # mape_lower = mean_absolute_percentage_error(y_test_lower, y_pred_lower)
# # print(f"Mean Squared Error for lower_bound: {mse_lower}")
# # print(f"Root Mean Squared Error for lower_bound: {rmse_lower}")
# # print(
# #     f"Normalised Root Mean Squared Error for lower_bound: {rmse_lower / y_test_lower.mean()}"
# # )
# # print(f"Mean Absolute Percentage Error for lower_bound: {mape_lower}")


# print("Upper Bound")
# # print("Linear Regression")

# # Initialize and train the model for upper_bound
# model_upper = LinearRegression()
# model_upper.fit(X_train, y_train_upper)

# # Predict and evaluate the model for upper_bound
# y_pred_upper = model_upper.predict(X_test)
# print(y_test_upper.head(), pd.DataFrame(y_pred_upper).head())

# mse_upper = mean_squared_error(y_test_upper, y_pred_upper)
# rmse_upper = root_mean_squared_error(y_test_upper, y_pred_upper)
# mape_upper = mean_absolute_percentage_error(y_test_upper, y_pred_upper)
# print(f"Mean Squared Error for upper_bound: {mse_upper}")
# print(f"Root Mean Squared Error for upper_bound: {rmse_upper}")
# print(
#     f"Normalised Root Mean Squared Error for upper_bound: {rmse_upper / y_test_upper.mean()}"
# )
# print(f"Mean Absolute Percentage Error for upper_bound: {mape_upper}")

# # print("XGBoost Regression")

# # model_upper = XGBRegressor(objective="reg:squarederror")
# # model_upper.fit(X_train, y_train_upper)

# # y_pred_upper = model_upper.predict(X_test)
# # mse_upper = mean_squared_error(y_test_upper, y_pred_upper)
# # rmse_upper = root_mean_squared_error(y_test_upper, y_pred_upper)
# # mape_upper = mean_absolute_percentage_error(y_test_upper, y_pred_upper)
# # print(f"Mean Squared Error for upper_bound: {mse_upper}")
# # print(f"Root Mean Squared Error for upper_bound: {rmse_upper}")
# # print(
# #     f"Normalised Root Mean Squared Error for upper_bound: {rmse_upper / y_test_upper.mean()}"
# # )
# # print(f"Mean Absolute Percentage Error for upper_bound: {mape_upper}")

# print()
# print("-" * 50)
# print()

print("Regression on Raw MI")

X = df.drop(columns=["lower_bound", "upper_bound"])
# X = (prepare_time_series_data(df, ["lower_bound", "upper_bound"], 10).shape)

y_lower = df["lower_bound_target"]
y_upper = df["upper_bound_target"]

# Split the data into training and testing sets
X_train, X_test, y_train_lower, y_test_lower, y_train_upper, y_test_upper = (
    train_test_split(X, y_lower, y_upper, test_size=float(sys.argv[3]), random_state=42)
)


print("Lower Bound")


# print("Linear Regression")
# Initialize and train the model for lower_bound
model_lower = LinearRegression()
model_lower.fit(X_train, y_train_lower)

# Predict and evaluate the model for lower_bound
y_pred_lower = model_lower.predict(X_test)

print(y_test_lower.head(), pd.DataFrame(y_pred_lower).head())
mse_lower = mean_squared_error(y_test_lower, y_pred_lower)
rmse_lower = root_mean_squared_error(y_test_lower, y_pred_lower)
mape_lower = mean_absolute_percentage_error(y_test_lower, y_pred_lower)
print(f"Mean Squared Error for lower_bound: {mse_lower}")
print(f"Root Mean Squared Error for lower_bound: {rmse_lower}")
print(
    f"Normalised Root Mean Squared Error for lower_bound: {rmse_lower / y_test_lower.mean()}"
)
print(f"Mean Absolute Percentage Error for lower_bound: {mape_lower}")

# print("XGBoost Regression")

# model_lower = XGBRegressor(objective="reg:squarederror")
# model_lower.fit(X_train, y_train_lower)

# y_pred_lower = model_lower.predict(X_test)
# mse_lower = mean_squared_error(y_test_lower, y_pred_lower)
# rmse_lower = root_mean_squared_error(y_test_lower, y_pred_lower)
# mape_lower = mean_absolute_percentage_error(y_test_lower, y_pred_lower)
# print(f"Mean Squared Error for lower_bound: {mse_lower}")
# print(f"Root Mean Squared Error for lower_bound: {rmse_lower}")
# print(
#     f"Normalised Root Mean Squared Error for lower_bound: {rmse_lower / y_test_lower.mean()}"
# )
# print(f"Mean Absolute Percentage Error for lower_bound: {mape_lower}")


print("Upper Bound")
# print("Linear Regression")

# Initialize and train the model for upper_bound
model_upper = LinearRegression()
model_upper.fit(X_train, y_train_upper)

# Predict and evaluate the model for upper_bound
y_pred_upper = model_upper.predict(X_test)
print(y_test_upper.head(), pd.DataFrame(y_pred_upper).head())

mse_upper = mean_squared_error(y_test_upper, y_pred_upper)
rmse_upper = root_mean_squared_error(y_test_upper, y_pred_upper)
mape_upper = mean_absolute_percentage_error(y_test_upper, y_pred_upper)
print(f"Mean Squared Error for upper_bound: {mse_upper}")
print(f"Root Mean Squared Error for upper_bound: {rmse_upper}")
print(
    f"Normalised Root Mean Squared Error for upper_bound: {rmse_upper / y_test_upper.mean()}"
)
print(f"Mean Absolute Percentage Error for upper_bound: {mape_upper}")

# print("XGBoost Regression")

# model_upper = XGBRegressor(objective="reg:squarederror")
# model_upper.fit(X_train, y_train_upper)

# y_pred_upper = model_upper.predict(X_test)
# mse_upper = mean_squared_error(y_test_upper, y_pred_upper)
# rmse_upper = root_mean_squared_error(y_test_upper, y_pred_upper)
# mape_upper = mean_absolute_percentage_error(y_test_upper, y_pred_upper)
# print(f"Mean Squared Error for upper_bound: {mse_upper}")
# print(f"Root Mean Squared Error for upper_bound: {rmse_upper}")
# print(
#     f"Normalised Root Mean Squared Error for upper_bound: {rmse_upper / y_test_upper.mean()}"
# )
# print(f"Mean Absolute Percentage Error for upper_bound: {mape_upper}")
