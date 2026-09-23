'''
Deep Long Short-Term Memory (LSTM) (concise implementation)
'''

from torch import nn
from dl_utils.runtime.devices import try_gpu
from dl_utils.d2l.rnn import RNNModel, train_ch8
from dl_utils.d2l.time_machine import load_data_time_machine

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device, net_name='Double-layer LSTM (concise)')

# plt.ioff()
# plt.show()
