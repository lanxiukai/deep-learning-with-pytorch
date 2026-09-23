"""Dataset-level accuracy and loss evaluation for supervised lessons."""

import torch
from torch import nn

from dl_utils.training.metrics import Accumulator


def accuracy(predictions, targets):
    """
    Compute the number of correct predictions.

    Args:
        predictions: the predicted value (batch_size, num_outputs) or (batch_size,)
        targets: the true value (batch_size,)
    Returns:
        the number of correct predictions
    """
    # A second dimension with multiple entries means predictions is a matrix.
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predicted_indices = predictions.argmax(
            axis=1
        )  # (batch_size,), index of the maximum probability
        matches = predicted_indices.type(targets.dtype) == targets  # True or False
        return float(matches.type(targets.dtype).sum())  # sum True values as a float
    return 0.0  # return 0.0 if predictions is not a matrix


def evaluate_accuracy(net, data_iter):
    """
    Compute the accuracy for a model on a dataset.

    Args:
        net: the network
        data_iter: the data iterator
    Returns:
        the accuracy of the model
    """
    if isinstance(
        net, torch.nn.Module
    ):  # Determine whether net is an instance of torch.nn.Module
        net.eval()  # set the model to evaluation mode
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for features, labels in data_iter:
            metric.add(
                accuracy(net(features), labels), labels.numel()
            )  # metric.add(correct predictions, total predictions)
    return metric[0] / metric[1]  # return the accuracy


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    Evaluate the accuracy of the model on the given dataset using GPU.

    Args:
        net: the model
        data_iter: the data iterator
        device: the device to use (Default: None)
    Returns:
        The accuracy of the model on the given dataset using GPU
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            # Get the device of the first parameter of the net
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # correct predictions, total predictions
    with torch.no_grad():
        for features, labels in data_iter:
            if isinstance(features, list):
                # Required for BERT fine-tuning (to be introduced later)
                device_features = [feature.to(device) for feature in features]
            else:
                device_features = features.to(device)
            device_labels = labels.to(device)
            metric.add(
                accuracy(net(device_features), device_labels), device_labels.numel()
            )
    return metric[0] / metric[1]  # Return the accuracy


def evaluate_loss(net, data_iter, loss):
    """
    Evaluate the model's loss on the given dataset.

    Args:
        net: the model
        data_iter: the data iterator
        loss: the loss function
    Returns:
        the average loss
    """
    metric = Accumulator(2)  # loss_sum, num_samples
    for features, labels in data_iter:
        out = net(features)
        labels = labels.reshape(out.shape)
        batch_loss = loss(out, labels)
        metric.add(batch_loss.sum(), batch_loss.numel())
    return metric[0] / metric[1]
