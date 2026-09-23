"""
Attention Scoring Function
"""

import torch
from dl_utils.plot._backend import pyplot as plt
from dl_utils.d2l.attention import AdditiveAttention, DotProductAttention
from dl_utils.plot.figures import heatmap

queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# Mini-batch of values; the two value matrices are identical.
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                  dropout=0.1)
attention.eval()
print(attention(queries, keys, values, valid_lens))
print(attention(queries, keys, values, valid_lens).shape)  # (2, 1, 4)

heatmap(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens))
print(attention(queries, keys, values, valid_lens).shape)  # (2, 1, 4)

heatmap(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

# plt.ioff()
# plt.show()
