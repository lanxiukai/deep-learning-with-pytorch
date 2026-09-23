'''
Sequence to Sequence (Seq2Seq)
'''

import torch
from torch import nn
from dl_utils.runtime.devices import try_gpu
from dl_utils.d2l.seq2seq import (
    Decoder,
    Encoder,
    EncoderDecoder,
    MaskedSoftmaxCELoss,
    bleu,
    predict_seq2seq,
    sequence_mask,
    train_seq2seq,
)
from dl_utils.d2l.translation import load_data_nmt

#@save
class Seq2SeqEncoder(Encoder):
    """Recurrent neural network encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # Input 'X' shape: (batch_size, num_steps)
        X = self.embedding(X)   # Output 'X' shape: (batch_size, num_steps, embed_size)
        # In recurrent neural network models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)  # Permute the dimensions of 'X' to (num_steps, batch_size, embed_size)
        output, state = self.rnn(X)
        # if state is not specified, it defaults to zeros
        # Shape of output: (num_steps, batch_size, num_hiddens)
        # Shape of state: (num_layers, batch_size, num_hiddens)
        return output, state

class Seq2SeqDecoder(Decoder):
    """Recurrent neural network decoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)  # Permute the dimensions of 'X' to (num_steps, batch_size, embed_size)
        # Broadcast context so that it has the same num_steps as X,
        # Repeating the state of the last layer (state[-1], shape: (batch_size, num_hiddens)) for each time step
        context = state[-1].repeat(X.shape[0], 1, 1)  # Shape of context: (num_steps, batch_size, num_hiddens)
        X_and_context = torch.cat((X, context), 2)
        # Shape of X_and_context: (num_steps, batch_size, embed_size + num_hiddens)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # Shape of output: (batch_size, num_steps, vocab_size)
        # Shape of state: (num_layers, batch_size, num_hiddens)
        return output, state

def main():
    # encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # encoder.eval()
    # X = torch.zeros((4, 7), dtype=torch.long)
    # output, state = encoder(X)
    # print(output.shape)
    # print(state.shape)

    # decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    # decoder.eval()
    # state = decoder.init_state(encoder(X))
    # output, state = decoder(X, state)
    # print(output.shape, state.shape)

    # X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # print(sequence_mask(X, torch.tensor([1, 2])))

    # X = torch.ones(2, 3, 4)
    # print(sequence_mask(X, torch.tensor([1, 2]), value=-1))

    # loss = MaskedSoftmaxCELoss()
    # print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
    #      torch.tensor([4, 2, 0])))

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                            dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                            dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device, 
                  net_name='Seq2Seq-GRU')

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

if __name__ == '__main__':
    main()
