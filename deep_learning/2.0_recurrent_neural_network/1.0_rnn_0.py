'''
Recurrent Neural Networks
'''

import torch
from dl_utils.runtime.devices import try_gpu
from dl_utils.d2l.rnn import RNNModelScratch, train_ch8
from dl_utils.d2l.time_machine import load_data_time_machine

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Parameters for the hidden layer
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # Parameters for the output layer
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    # Initialize the hidden state of the RNN to zero
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    # Shape of inputs: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    hidden_state, = state
    outputs = []
    # Shape of X: (batch_size, vocab_size)
    for input_step in inputs:
        hidden_state = torch.tanh(
            torch.mm(input_step, W_xh) + torch.mm(hidden_state, W_hh) + b_h)
        output_step = torch.mm(hidden_state, W_hq) + b_q
        outputs.append(output_step)
    # outputs: (num_steps * batch_size, vocab_size), state: (batch_size, num_hiddens)
    return torch.cat(outputs, dim=0), (hidden_state,)

def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    # print(F.one_hot(torch.tensor([0, 2]), len(vocab)))  # one-hot encoding, (2, 28)
    # X = torch.arange(10).reshape((2, 5))
    # print(F.one_hot(X.T, 28).shape)  # (5, 2, 28)

    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(),
                          get_params, init_rnn_state, rnn)
    # state = net.begin_state(X.shape[0], try_gpu())
    # Y, new_state = net(X.to(try_gpu()), state)
    # print(Y.shape, len(new_state), new_state[0].shape)
    # (num_steps * batch_size, vocab_size), 1, (batch_size, num_hiddens)

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu(),
              net_name='RNN Scratch (sequential partitioning)')

    train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu(),
              use_random_iter=True, net_name='RNN Scratch (random sampling)')
    
    # plt.ioff()
    # plt.show()

if __name__ == '__main__':
    main()
