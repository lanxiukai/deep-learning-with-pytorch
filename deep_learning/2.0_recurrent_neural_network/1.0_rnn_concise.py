'''
Recurrent Neural Networks (Concise Implementation)
'''

from torch import nn
from dl_utils.runtime.devices import try_gpu
from dl_utils.d2l.rnn import RNNModel, train_ch8
from dl_utils.d2l.time_machine import load_data_time_machine

def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    # shape of state: (num_layers, batch_size, num_hiddens)
    # state = torch.zeros((1, batch_size, num_hiddens))
    # print(state.shape)  # (1, 32, 256)

    # X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    # Y, state_new = rnn_layer(X, state)
    # print(Y.shape, state_new.shape)  # (35, 32, 256), (1, 32, 256)

    device = try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs,
                       device, net_name='RNN (concise)')

    # plt.ioff()
    # plt.show()

if __name__ == '__main__':
    main()
