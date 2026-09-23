import math

import torch
from torch import nn
from torch.nn import functional as F

from dl_utils.plot.figures import Animator
from dl_utils.training.metrics import Accumulator
from dl_utils.training.optimization import grad_clipping, sgd
from dl_utils.training.timing import Timer

class RNNModelScratch:
    """
    Recurrent neural network model implemented from scratch (for Chapter 8).

    Args:
        vocab_size: the size of the vocabulary
        num_hiddens: the number of hidden units
        device: the device to use
        get_params: a function to get the parameters
        init_state: a function to initialize the state
        forward_fn: a function to forward the model
    """
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """
        Forward pass (allow the object to be called like a function).

        Args:
            X: the input features (batch_size, num_steps)
            state: the state (num_layers, batch_size, num_hiddens) if LSTM, otherwise (batch_size, num_hiddens)
        """
        inputs = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # X: (num_steps, batch_size, vocab_size)
        return self.forward_fn(inputs, state, self.params)

    def begin_state(self, batch_size, device):
        """
        Initialize the state.

        Args:
            batch_size: the batch size
            device: the device to use
        """
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    Generate new characters following the given prefix.

    Args:
        prefix: a string of the prefix
        num_preds: the number of predictions
        net: the network
        vocab: a vocabulary object
        device: the device to use
    Returns:
        A string of the generated characters
    """
    state = net.begin_state(batch_size=1, device=device)  # (1, num_hiddens)
    outputs = [vocab[prefix[0]]]  # [index] of the prefix and generated characters
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # [last index] -> (1, 1): (num_steps, batch_size)
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)  # warm-up state only
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict for num_preds steps
        y, state = net(get_input(), state)  # (1 * 1, vocab_size), (1, num_hiddens)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_epoch_ch8(
        net: "RNNModelScratch | RNNModel",
        train_iter, loss, updater, device,
        use_random_iter, timer):
    """
    Train the network for one epoch (see Chapter 8 for the definition).

    Args:
        net: the network
        train_iter: the training data iterator
        loss: the loss function
        updater: the optimizer
        device: the device to use
        use_random_iter: whether to use random sampling
        timer: the timer instance (Timer)
    Returns:
        A tuple of perplexity and the number of tokens: (ppl, num_tokens)
        - ppl: the perplexity
        - num_tokens: the number of tokens (num_steps * batch_size)
    """
    state = None
    metric = Accumulator(2)  # l.sum(), num_tokens (num_steps * batch_size)
    for X, Y in train_iter:
        timer.start()
        if state is None or use_random_iter:
            # Initialize state during the first iteration or when using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # detach the state from the computation graph, avoid gradient explosion
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # For nn.GRU, state is a tensor
                state.detach_()
            else:
                # For nn.LSTM or for our scratch implementation, state is a tuple of tensors
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)  # Flatten over num_steps (num_steps * batch_size)
        X_device, y = X.to(device), y.to(device)
        y_hat, state = net(X_device, state)  # (num_steps * batch_size, vocab_size), (batch_size, num_hiddens)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
        timer.stop()
    if metric[1] == 0:
        return math.inf, 0
    return math.exp(metric[0] / metric[1]), metric[1]  # perplexity, num_tokens


def train_ch8(
        net: "RNNModelScratch | RNNModel",
        train_iter, vocab, lr, num_epochs, device,
        use_random_iter=False, net_name=None):
    """
    Train the model (see Chapter 8 for the definition).

    Args:
        net: the network
        train_iter: the training data iterator
        vocab: a vocabulary object
        lr: the learning rate
        num_epochs: the number of epochs
        device: the device to use
        use_random_iter: whether to use random sampling (Default: False)
        net_name: the name of the network (Default: None)
    Returns:
        None: prints the perplexity, speed, and total time, and the generated characters
    """
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialization
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    if net_name is not None:
        print(f'\n{net_name} is training on {str(device)} ...')
    else:
        print(f'\nTraining on {str(device)} ...')

    timer, total_tokens = Timer(), 0.0
    ppl = None
    # Training and prediction
    for epoch in range(num_epochs):
        ppl, num_tokens = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter, timer)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
        total_tokens += num_tokens
    if ppl is None:
        return
    elapsed = timer.sum()
    tokens_per_sec = total_tokens / elapsed if elapsed else 0.0
    print(f'Perplexity {ppl:.1f}, {tokens_per_sec:.1f} tokens/sec, total time: {timer.format_time()}')
    print(predict('time traveller'))
    print(predict('traveller'))


class RNNModel(nn.Module):
    """
    Recurrent neural network model (for Chapter 8).

    Args:
        rnn_layer: the RNN layer
        vocab_size: the size of the vocabulary
        **kwargs: additional arguments
    """
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (introduced later),
        # num_directions should be 2; otherwise it should be 1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        """
        Forward pass.

        Args:
            inputs: the input features (batch_size, num_steps)
            state: the state (num_directions * num_layers, batch_size, num_hiddens)
        Returns:
            A tuple of the output and the state: (output, state)
            - output: the output (num_steps * batch_size, vocab_size)
            - state: the state (num_directions * num_layers, batch_size, num_hiddens)
        """
        X_one_hot = F.one_hot(inputs.T.long(), self.vocab_size)  # X: (num_steps, batch_size, vocab_size)
        X_float = X_one_hot.to(torch.float32)
        Y, state = self.rnn(X_float, state)  # Y: (num_steps, batch_size, num_hiddens * num_directions)
        # First reshape Y to (num_steps * batch_size, num_hiddens * num_directions)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        """
        Initialize the state.

        Args:
            device: the device to use
            batch_size: the batch size (Default: 1)
        Returns:
            A tuple of the state: (H, C) if LSTM, otherwise (H,)
            - H: (num_directions * num_layers, batch_size, num_hiddens)
            - C: (num_directions * num_layers, batch_size, num_hiddens) if LSTM, otherwise None
        """
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU uses a tensor (H,) as its hidden state
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM uses a tuple of tensors (H, C) as its hidden state
            return (torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
