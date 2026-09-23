'''
Bidirectional Long Short-Term Memory (LSTM) (concise implementation)
'''

from torch import nn
from dl_utils.runtime.devices import try_gpu
from dl_utils.d2l.rnn import RNNModel, train_ch8
from dl_utils.d2l.time_machine import load_data_time_machine

# Load data
batch_size, num_steps, device = 32, 35, try_gpu()
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

# Define the bidirectional LSTM model by setting "bidirectional=True"
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2  # each layer has two directions
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = RNNModel(lstm_layer, len(vocab))
model = model.to(device)

# Train the model
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, device, net_name='Double-layer Bi-LSTM (concise)')

# plt.ioff()
# plt.show()
