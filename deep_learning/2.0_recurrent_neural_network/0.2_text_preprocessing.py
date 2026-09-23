'''
Text Preprocessing
'''

from dl_utils.d2l.time_machine import load_corpus_time_machine, read_time_machine
from dl_utils.d2l.vocabulary import Vocab, count_corpus, tokenize

lines = read_time_machine()
print(f'# The total number of text lines: {len(lines)}')
print(lines[0])
print(lines[10])

tokens = tokenize(lines) or []
for i in range(11):
    print(tokens[i])
print('--------------------------------')

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

for i in [0, 10]:
    print('Text:', tokens[i])
    print('Indices:', vocab[tokens[i]])

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))
print(vocab.idx_to_token[:10])
