'''
Language Models and Dataset
'''

from dl_utils.plot.figures import plot
from dl_utils.d2l.time_machine import (
    load_data_time_machine,
    read_time_machine,
    seq_data_iter_random,
    seq_data_iter_sequential,
)
from dl_utils.d2l.vocabulary import Vocab, tokenize

tokens = tokenize(read_time_machine()) or []
# Because each line of text is not necessarily a sentence or a paragraph
# we concatenate all the lines into a single list
corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus)
print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
     xscale='log', yscale='log')

bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
     ylabel='frequency: n(x)', xscale='log', yscale='log',
     legend=['unigram', 'bigram', 'trigram'])

my_seq = list(range(35))
for features, labels in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', features, '\nY:', labels)

for features, labels in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', features, '\nY:', labels)

    # plt.ioff()
    # plt.show()
