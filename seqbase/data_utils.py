import json 
import os
import csv
import pandas as pd
import re
import torch
import unicodedata
import itertools
import random
import config


# Default word tokens
PAD_token = config.PAD_token  # Used for padding short sentences
SOS_token = config.SOS_token  # Start-of-sentence token
EOS_token = config.EOS_token  # End-of-sentence token
UNK_token = config.UNK_token  # Unkonw token


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"can't", r"can not", s)
    s = re.sub(r"n't", r" not", s)
    s = re.sub(r"'ve'", r" have", s)
    s = re.sub(r"cannot", r"can not", s)
    s = re.sub(r"what's", r"what is", s)
    s = re.sub(r"that's",r"that is",s)
    s = re.sub(r"'re", r" are", s)
    s = re.sub(r"'d", r" would", s)
    s = re.sub(r"'ll'", r" will", s)
    s = re.sub(r" im ", r" i am ", s)
    s = re.sub(r"'m", r" am", s)
    s = re.sub(r"([.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s



def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] if word in voc.word2index else UNK_token for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

                                       