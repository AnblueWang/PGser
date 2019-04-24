import json 
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import pandas as pd
import re
import unicodedata
import itertools
import random
import config


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size,hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.embedding_size = embedding_size

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq) # (L,B,E)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)# (L,B,2*H)  (layer*2,B,H)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs) 
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] #(L,B,H)
        # Return output and final hidden state
        return outputs, hidden

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output): #(1,B,H) (L,B,H)
        return torch.sum(hidden * encoder_output, dim=2) #(L,B)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t() #(B,L)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1) #(B,1,L)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, embedding_size, encoder_n_layers, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.encoder_n_layers = encoder_n_layers

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size+hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.concatsec = nn.Linear(hidden_size*2*encoder_n_layers,hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, sec_hidden, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step) #(1,B,E) 
        embedded = self.embedding_dropout(embedded)
        
        sec_hidden = sec_hidden.transpose(0,1).contiguous().view(1,embedded.shape[1],-1) #(1,B,2*encoderlayer*H)
        sec_hidden = self.concatsec(sec_hidden) #(1,B,H)
        # Forward through unidirectional GRU
        rnn_input = torch.cat((embedded,sec_hidden),2) #融合section与word (1,B,E+H)
        rnn_output, hidden = self.gru(rnn_input, last_hidden) #(1,B,H) (layer,B,H)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs) #(B,1,L)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))# (B,1,L) * (B,L,H) = (B,1,H)
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0) #(B,H)
        context = context.squeeze(1) #(B,H)
        concat_input = torch.cat((rnn_output, context), 1) #(B,2*H)
        concat_output = torch.tanh(self.concat(concat_input)) #(B,H)
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)#(B,Out)
        output = F.softmax(output, dim=1)#(B,Out)
        # Return output and final hidden state
        return output, hidden #(B,Out) (layer,B,H)