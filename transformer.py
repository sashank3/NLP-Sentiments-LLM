import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

"""
This is a supplemental file to be run with train_transformer.ipynb.

Resources used to help devlop this mode include a tutorial "Building a Transformer with PyTorch" by Arjun Sarkar [3] 
as well as documentation from pytorch [2].   
"""


class PositionWiseFeedForward(nn.Module):
    """
    Passes the data through multiple linear layers in the network and applies a Relu activation function to the outputs.
    """
    def __init__(self, d_model, d_ff, layer_count=2):
        super(PositionWiseFeedForward, self).__init__()
        self.fcs = [nn.Linear(d_model, d_ff)]

        for i in range(layer_count):
            self.fcs.append(nn.Linear(d_ff, d_ff))

        self.fcs.append(nn.Linear(d_ff, d_model))

        self.relu = nn.ReLU()

    def forward(self, x):
        value = x
        for i in range(len(self.fcs) - 1):
            value = self.relu(self.fcs[i](value))

        return self.fcs[len(self.fcs) - 1](value)


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the encoding vectors. It does this by appending a vector of alternating sine/cosine
    values based on the index of each token.

    Based on Positional Encoding implementation present in "Language Modeling with NN.Transformer and TORCHTEXT" from
    the PyTorch documentation [2].
    """

    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting a self attention layer that feeds into a feed forward layer.
    """

    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        attention_output = self.self_attn(x, x, x)[0]
        x = self.norm1(x + attention_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class Transformer(nn.Module):
    """
    The encoder-only transformer model. Input is passed to an embedding layer followed by a positional encoder layer.
    The value is then passed through the encoder layers. The output of the encoder is then passed to a linear layer
    which aggregates to a single output. The output is then passed through a sigmoid layer to get the
    final classification value.
    """

    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model * max_seq_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        """
        On the forward pass, Finally, the encoder output is flattened and passed to linear layer, which produces a
        single classification value that is normalized after being passed through a sigmoid.
        :param src: data to classify
        :return: the normalized output
        """
        src_embedded = self.positional_encoding(self.encoder_embedding(src))

        encoder_output = src_embedded
        for enc_layer in self.encoder_layers:
            encoder_output = enc_layer(encoder_output)

        output = self.fc(encoder_output.reshape(encoder_output.shape[0], -1))
        normalized_output = self.sigmoid(output)

        return normalized_output
