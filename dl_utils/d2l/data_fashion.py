import torch

from dl_utils.data.vision import vision_loaders
from dl_utils.filesystem.project_root import infer_project_root
from dl_utils.plot.figures import Animator
from dl_utils.plot.images import show_images
from dl_utils.training.metrics import Accumulator, accuracy, evaluate_accuracy


PROJECT_ROOT = infer_project_root()


def get_fashion_mnist_labels(labels):
    """
    Return text labels for the Fashion-MNIST dataset.

    Args:
        labels: the labels
    Returns:
        the text labels (1D list [text_label])
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size, resize=None, process_count=4):
    """
    Download the Fashion-MNIST dataset and load it into memory.

    Delegates to ``vision_loaders`` for unified dataset handling.

    Args:
        batch_size:    the batch size
        resize:        optional resize for image inputs (Default: None)
        process_count: number of DataLoader worker processes (Default: 4)

    Returns:
        A tuple of (train_iter, test_iter) ``DataLoader`` objects.
    """
    return vision_loaders(
        'fashion_mnist',
        data_dir=PROJECT_ROOT / 'data' / 'fashion_mnist',
        batch_size=batch_size,
        resize=resize,
        num_workers=process_count,
    )


def train_epoch_ch3(net, train_iter, loss, updater):
    """
    Train the model for one epoch.

    Args:
        net: the network
        train_iter: the training data iterator
        loss: the loss function
        updater: the optimizer
    Returns:
        A tuple: (average_loss, average_accuracy)
    """
    if isinstance(net, torch.nn.Module): # Determine whether net is an instance of torch.nn.Module
        net.train()  # set the model to training mode
    metric = Accumulator(3)  # l.sum(), correct predictions, num_samples
    for X, y in train_iter:
        # compute the gradient and update the parameters
        y_hat = net(X)      # y_hat: the predicted value (batch_size, num_outputs) or (batch_size,)
        l = loss(y_hat, y)  # l: the loss (batch_size,); y: the true value (batch_size,)
        if isinstance(updater, torch.optim.Optimizer):  # Determine whether updater is an instance of torch.optim.Optimizer
            # use the built-in optimizer and loss function in PyTorch
            updater.zero_grad()  # reset the gradient
            l.mean().backward()  # compute the gradient
            updater.step()       # update the parameters
        else:
            # use the custom optimizer and loss function
            l.sum().backward()  # compute the gradient
            updater(X.shape[0])  # update the parameters
        metric.add(l.detach().sum().item(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss,
              num_epochs, updater):
    """
    Train a model with multiple epochs.

    Args:
        net: the network
        train_iter: the training data iterator
        test_iter: the test data iterator
        loss: the loss function
        num_epochs: the number of epochs
        updater: the optimizer
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    train_metrics = None
    test_acc = None
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # train_metrics: (train_loss, train_acc)
        test_acc = evaluate_accuracy(net, test_iter)                     # test_acc: the accuracy of the test data
        animator.add(epoch + 1, train_metrics + (test_acc,))             # add the train_metrics and test_acc to the animator
    if train_metrics is None or test_acc is None:
        return
    train_loss, train_acc = train_metrics
    # Check whether the data is out of bounds
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def predict_ch3(net, test_iter, n=6):
    """
    Predict labels.

    Args:
        net: the network
        test_iter: the test data iterator
        n: the number of images to predict (Default: 6)
    """
    for X, y in test_iter:
        break
    else:
        return
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
