'''
Gated Recurrent Units (GRU) (Concise Implementation)
'''

from torch import nn
from dl_utils.runtime.devices import try_gpu
from dl_utils.d2l.rnn import RNNModel, train_ch8
from dl_utils.d2l.time_machine import load_data_time_machine

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, try_gpu()
num_epochs, lr = 500, 1

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = RNNModel(gru_layer, len(vocab))
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device,
                   net_name='GRU (concise)')

# plt.ioff()
# plt.show()
