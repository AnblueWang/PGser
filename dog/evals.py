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
from data_utils import indexesFromSentence,normalizeString

def evaluate(encoder, sec_encoder, decoder, searcher, voc, sentence, wikiSec):
    max_length=config.MAX_LENGTH
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)] #(1,L)
    sec_indexes = [indexesFromSentence(voc,wikiSec)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch]) #(1,)
    sec_lengths = torch.tensor([len(indexes) for indexes in sec_indexes])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1) #(L,1)
    sec_batch = torch.LongTensor(sec_indexes).transpose(0, 1) 
    # Use appropriate device
    input_batch = input_batch.to(config.device)
    sec_batch = sec_batch.to(config.device)
    lengths = lengths.to(config.device)
    sec_lengths = sec_lengths.to(config.device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, sec_batch, sec_lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token] for token in tokens if not (token == config.EOS_token or token == config.PAD_token)]
    return decoded_words


def evaluateInput(encoder, sec_encoder, decoder, searcher, voc, wiki_strings):
    input_sentence = ''
    while(1):
        try:
            doc_idx = int(input('document index:'))
            sec_idx = int(input('section index:'))
            sec_sentence = wiki_strings[doc_idx][sec_idx]
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, sec_encoder, decoder, searcher, voc, input_sentence, sec_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words ]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
