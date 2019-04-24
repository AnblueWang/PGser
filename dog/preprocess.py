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

def readConvsFile(path,file):
    conv_file = open(os.path.join(conv_path,file))
    conv_data = json.load(conv_file)
    wikiIndex = conv_data['wikiDocumentIdx']
    if len(conv_data['whoSawDoc']) == 2:
        saw = 2
    elif conv_data['whoSawDoc'] == ['user1']:
        saw = 0
    else:
        saw = 1
    convs = []
    history = ''
    for idx, utter in enumerate(conv_data['history']):
        utter['text'] = normalizeString(utter['text'])
        line = {}
        line['wikiIdx'] = wikiIndex
        line['docIdx'] = utter['docIdx']
        line['uid'] = utter['uid']
        line['history'] = history
        line['response'] = utter['text']
        history += utter['text']+' '
        line['saw'] = saw
        convs.append(line)
    
    return convs

def saveNewConvs(read_path,save_path):
    index = 0
    for file in os.listdir(read_path):
        if file.split('.')[1] != 'json':
            continue
        convs = readConvsFile(read_path,file)
        new_file = 'train'+str(index)+'.csv'
        data_file = os.path.join(save_path,new_file)
        print('Writing to new formatted line...',index)
        df = pd.DataFrame(convs)
        df.to_csv(data_file,encoding='utf-8',sep='\t')
        index += 1
        
if __name__ == '__main__':
    read_path = '../Conversations/train'
    save_path = '../Conversations/seq+att/train'
    saveNewConvs(read_path,save_path) 
