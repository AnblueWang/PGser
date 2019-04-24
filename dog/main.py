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
from model import EncoderRNN,LuongAttnDecoderRNN 
from decoder import GreedySearchDecoder,BeamSearchDecoder
from train import train, trainIters 
from data_utils import normalizeString,loadPrepareData,trimRareWords,process_wiki_document
from evals import evaluateInput
import config



#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


wiki_strings = process_wiki_document('../WikiData')

corpus_name = 'seq+att'
save_dir = os.path.join('../Conversations',corpus_name)
voc, pairs = loadPrepareData(config.data_path,corpus_name,wiki_strings)

pairs,wiki_strings = trimRareWords(voc, pairs, config.MIN_COUNT,wiki_strings)
# print("\npairs:")
# for pair in pairs[:30]:
#     print(pair)
# print(wiki_strings[0])


# Load model if a loadFilename is provided
if config.loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(config.loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    sec_encoder_sd = checkpoint['sec_en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    sec_encoder_optimizer_sd = checkpoint['sec_en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, config.embedding_size)
if config.loadFilename:
    embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models
encoder = EncoderRNN(config.embedding_size,config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
sec_encoder = EncoderRNN(config.embedding_size,config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
decoder = LuongAttnDecoderRNN(config.attn_model, embedding,config.embedding_size, config.encoder_n_layers,config.hidden_size, voc.num_words, config.decoder_n_layers, config.dropout)

if config.loadFilename:
    encoder.load_state_dict(encoder_sd)
    sec_encoder.load_state_dict(sec_encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(config.device)
sec_encoder = sec_encoder.to(config.device)
decoder = decoder.to(config.device)
print('Models built and ready to go!')


# Ensure dropout layers are in train mode
encoder.train()
sec_encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
sec_encoder_optimizer = optim.Adam(sec_encoder.parameters(), lr=config.learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * config.decoder_learning_ratio)
if config.loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    sec_encoder_optimizer.load_state_dict(sec_encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(voc, pairs, wiki_strings, encoder, sec_encoder,decoder, encoder_optimizer,sec_encoder_optimizer,decoder_optimizer,embedding,save_dir)

# Set dropout layers to eval mode
encoder.eval()
sec_encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, sec_encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, sec_encoder, decoder, searcher, voc,wiki_strings)