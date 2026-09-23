'''
Machine Translation and Dataset
'''

import torch
from dl_utils.plot.figures import seq_len_hist
from dl_utils.d2l.translation import (
    build_array_nmt,
    load_data_nmt,
    preprocess_nmt,
    read_data_nmt,
    tokenize_nmt,
    truncate_pad,
)
from dl_utils.d2l.vocabulary import Vocab

raw_text = read_data_nmt()
print(raw_text[:75])

text = preprocess_nmt(raw_text)
print(text[:80])

source, target = tokenize_nmt(text)
print('Source:', source[:6])
print('Target:', target[:6])

seq_len_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);

    # plt.ioff()
    # plt.show()

src_vocab = Vocab(source, min_freq=2,
                  reserved_tokens=['<pad>', '<bos>', '<eos>'])  # source vocabulary
print('Length of source vocabulary:', len(src_vocab))

answer = truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
print('Truncated and padded source sequence:', answer)

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=5, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('Valid length of X:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Valid length of Y:', Y_valid_len)
    break
