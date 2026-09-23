"""
Self-Attention and Positional Encoding
"""

import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.d2l.attention import MultiHeadAttention, PositionalEncoding
from dl_utils.plot.figures import heatmap, plot

num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
print(attention.eval())

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens).shape)

encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])

for i in range(8):
    print(f'The binary of {i} is: {i:>03b}')

P = P[0, :, :].unsqueeze(0).unsqueeze(0)
heatmap(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')

# plt.ioff()
# plt.show()
