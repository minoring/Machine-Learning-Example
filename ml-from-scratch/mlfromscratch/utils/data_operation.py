import numpy as np


def mean_squared_error(y_true, y_pred):
  """Returns the mean squared error between y_true and y_pred"""
  mse = np.mean(np.power(y_true - y_pred, 2))
  return mse


def accuracy_score(y_true, y_pred):
  """Compare y_true to y_pred and return the accuracy"""
  accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
  return accuracy
