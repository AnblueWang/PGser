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


# Default word tokens
PAD_token = config.PAD_token  # Used for padding short sentences
SOS_token = config.SOS_token  # Start-of-sentence token
EOS_token = config.EOS_token  # End-of-sentence token
UNK_token = config.UNK_token  # Unkonw token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token:"UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token, "UNK":UNK_token}
        self.word2count = {"UNK": 0}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token:"UNK"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)

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
    s = re.sub(r"'re", r" are", s)
    s = re.sub(r"'d", r" would", s)
    s = re.sub(r"'ll'", r" will", s)
    s = re.sub(r" im ", r" i am ", s)
    s = re.sub(r"'m", r" am", s)
    s = re.sub(r"([.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s



def buildPairs(df):
    pairs = []
    for i in df.index:

        pair = []
        pair.append(df.iloc[i].wikiIdx)
        pair.append(df.iloc[i].docIdx)
        if i > 0 and df.iloc[i].uid != df.iloc[i-1].uid:
            pair.append(df.iloc[i].history)
            pair.append(df.iloc[i].response)
            pairs.append(pair)
    return pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    try:
        # Input sequences need to preserve the last word for EOS token
        return type(p[2]) == str and type(p[3]) == str and len(p[2].split(' ')) < config.MAX_HISTORY_LENGTH and len(p[3].split(' ')) < config.MAX_LENGTH
    except Exception as e:
        print(p)
        raise e
        

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(dataPath,corpus_name,wiki_strings):
    voc = Voc(corpus_name)
    pairs = []
    print("Starting preparing training data...")
    for file in os.listdir(dataPath):
        df = pd.read_csv(os.path.join(dataPath,file),sep='\t',encoding='utf-8')
        pairs += buildPairs(df)

    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[2])
        voc.addSentence(pair[3])
    
    for wiki_doc in wiki_strings:
        for s in wiki_doc:
            voc.addSentence(s)
            
    print("Counted words:", voc.num_words)
    return voc, pairs

def trimRareWords(voc, pairs, min_count,wiki_docs):
    try:
        # Trim words used under the MIN_COUNT from the voc
        voc.trim(min_count)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for index,pair in enumerate(pairs):
            input_sentence = pair[2]
            output_sentence = pair[3]
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    input_sentence = re.sub(" "+word+" "," UNK ",input_sentence)
                    input_sentence = re.sub("^"+word+" ","UNK ",input_sentence)
                    input_sentence = re.sub(" "+word+"$"," UNK",input_sentence)
                    input_sentence = re.sub("^"+word+"$","UNK",input_sentence)
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    output_sentence = re.sub(" "+word+" "," UNK ",output_sentence)
                    output_sentence = re.sub("^"+word+" ","UNK ",output_sentence)
                    output_sentence = re.sub(" "+word+"$"," UNK",output_sentence)
                    output_sentence = re.sub("^"+word+"$","UNK",output_sentence)

            pairs[index][2] = input_sentence
            pairs[index][3] = output_sentence
        
        for index, wiki_doc in enumerate(wiki_docs):
            for si,section in enumerate(wiki_doc):
                for word in section.split(' '):
                    if word not in voc.word2index:
                        section = re.sub(" "+word+" "," UNK ", section)
                        section = re.sub("^"+word+" ","UNK ", section)
                        section = re.sub(" "+word+"$"," UNK", section)
                        section = re.sub("^"+word+"$","UNK", section)
                        wiki_docs[index][si] = section

        return pairs, wiki_docs
    except Exception as e:
        print(pair)
        raise e

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] if word in voc.word2index else UNK_token for word in sentence.split(' ')] + [EOS_token]

def process_wiki_document(path):
    wiki_docs = [0]*30
    for file in os.listdir(path):
        wiki_file = open(os.path.join(path,file))
        wiki_data = json.load(wiki_file)
        wiki_docs[wiki_data['wikiDocumentIdx']] = wiki_data

    wiki_strings = []
    for i in range(30):
        doc = []
        for j in range(4):
            doc.append(normalizeString(str(wiki_docs[i][str(j)])))
        wiki_strings.append(doc)
        
    return wiki_strings

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

def docPre(docs,voc):
    try:
        indexes_docs = []
        for doc in docs:
    #         print(doc)
            indexes_docs.append([indexesFromSentence(voc,sentence) for sentence in doc])
        return indexes_docs
    except Exception as e:
        print(doc)
        raise e

def docVar(l,voc,docs):
    indexes_docs = docPre(docs,voc)
    indexes_batch = [indexes_docs[docIdx][secIdx] for docIdx,secIdx in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths
    

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

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch,wiki_docs):
    try:
        pair_batch.sort(key=lambda x: len(x[2].split(" ")), reverse=True)
        doc_batch, input_batch, output_batch = [], [], []
        for pair in pair_batch:
            doc_batch.append([pair[0],pair[1]])
            input_batch.append(pair[2])
            output_batch.append(pair[3])
        doc_inp, doc_lengths = docVar(doc_batch, voc,wiki_docs)
        inp, lengths = inputVar(input_batch, voc)
        output, mask, max_target_len = outputVar(output_batch, voc)
        return doc_inp,doc_lengths,inp, lengths, output, mask, max_target_len
    except Exception as e:
        print(pair_batch)
        raise e

def batchGenerator(voc,pairs,wiki_strings):
    for _ in range(config.n_iteration):
        batch = batch2TrainData(voc, [random.choice(pairs) for _ in range(config.batch_size)],wiki_strings)
        yield batch
                                       