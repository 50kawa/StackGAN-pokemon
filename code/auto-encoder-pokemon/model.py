import random
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.hid_dim = cfg.CHARLSTM.DIMENSION
        self.n_layers = 1

        self.word_embeds = nn.Embedding(cfg.CHAR.VOCABSIZE, cfg.CHARVEC.DIMENSION)

        self.pokemon_embed = nn.Linear(cfg.POKEMON.SIZE, cfg.POKEMON.DIMENSION)

        self.rnn = nn.LSTM(cfg.CHARVEC.DIMENSION + cfg.POKEMON.DIMENSION, cfg.CHARLSTM.DIMENSION, 1, dropout=cfg.CHARLSTM.DROPOUT)

        self.dropout = nn.Dropout(cfg.CHARLSTM.DROPOUT)

    def forward(self, sentence, embedding):
        embeds = self.word_embeds(Variable(sentence))
        concated_emb = torch.cat((embeds, self.pokemon_embed(embedding).repeat(len(sentence), 1, 1)), dim=2)
        lstm_out, (hn, cn) = self.rnn(concated_emb)
        return hn, cn



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.output_dim = cfg.CHAR.VOCABSIZE
        self.hid_dim = cfg.CHARLSTM.DIMENSION
        self.n_layers = 1

        self.word_embeds = nn.Embedding(cfg.CHAR.VOCABSIZE + 1, cfg.CHARVEC.DIMENSION)

        self.pokemon_embed = nn.Linear(cfg.POKEMON.SIZE, cfg.POKEMON.DIMENSION)

        self.rnn = nn.LSTM(cfg.CHARVEC.DIMENSION + cfg.POKEMON.DIMENSION, cfg.CHARLSTM.DIMENSION, 1, dropout=cfg.CHARLSTM.DROPOUT)

        self.fc_out = nn.Linear(cfg.CHARLSTM.DIMENSION, cfg.CHAR.VOCABSIZE + cfg.POKEMON.SIZE)

        self.dropout = nn.Dropout(cfg.CHARLSTM.DROPOUT)

    def forward(self, sentence, embedding, hidden, cell):
        sentence = sentence.unsqueeze(0)
        embeds = self.word_embeds(Variable(sentence))
        concated_emb = torch.cat((embeds, self.pokemon_embed(embedding).unsqueeze(0)), dim=2)
        output, (hidden, cell) = self.rnn(concated_emb, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        output_word = prediction[:, :cfg.CHAR.VOCABSIZE]
        output_emb = prediction[:, cfg.CHAR.VOCABSIZE:]

        return output_word, output_emb, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.module.hid_dim == decoder.module.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.module.n_layers == decoder.module.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, pokemon_words, pokemon_emb, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # src = [src len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = pokemon_words.shape[1]
        src_len = pokemon_words.shape[0]
        src_vocab_size = self.decoder.module.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(src_len, batch_size, src_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(pokemon_words, pokemon_emb)

        # first input to the decoder is the <sos> tokens
        input = torch.tensor([cfg.CHAR.VOCABSIZE]).repeat(cfg.TRAIN.BATCH_SIZE)
        input_pokemon_emb = torch.zeros(cfg.TRAIN.BATCH_SIZE, cfg.POKEMON.SIZE)
        for t in range(src_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output_word, output_emb, hidden, cell = self.decoder(input, input_pokemon_emb, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output_word

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = torch.argmax(output_word, dim=1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if teacher_force:
                input = pokemon_words[t]
                input_pokemon_emb = pokemon_emb
            else:
                input = top1
                input_pokemon_emb = output_emb

        return outputs, output_emb