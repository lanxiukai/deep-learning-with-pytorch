import torch


def synthetic_data(w, b, num_examples):
    """
    Generate y = Xw + b + noise (gaussian noise, N(0, 0.01)).

    Args:
        w: the weights
        b: the bias
        num_examples: the number of examples
    Returns:
        A tuple of input features and the true values: (X, y)
        - X: the input features (num_examples, len(w))
        - y: the true values (num_examples, 1)
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def linreg(X, w, b):
    """
    The linear regression model.

    Args:
        X: the input features
        w: the weights
        b: the bias
    Returns:
        the predicted value (num_examples, 1)
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    Squared loss.

    Args:
        y_hat: the predicted value
        y: the true value
    Returns:
        the squared loss (multiply by 0.5 for convenience)
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
