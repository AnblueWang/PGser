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

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, sec_encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sec_encoder = sec_encoder

    def forward(self, input_seq, input_length, sec_seq, sec_length, max_length):
        # Forward input and section through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)#(L,1,H) (layer*direc,1,H)
        sec_outputs, sec_hidden = self.sec_encoder(sec_seq, sec_length)
        
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:config.decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=config.device, dtype=torch.long) * config.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=config.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=config.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder 
            decoder_output, decoder_hidden = self.decoder(decoder_input, sec_hidden, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens.tolist(), all_scores

class Beam(object):
    def __init__(self, tokens, log_probs, sec_hidden, decoder_hidden, encoder_outputs):
        self.tokens = tokens
        self.log_probs = log_probs
        self.sec_hidden = sec_hidden
        self.decoder_hidden = decoder_hidden
        self.encoder_outputs = encoder_outputs
        
    def extend(self, token, log_prob, sec_hidden, decoder_hidden, encoder_outputs):
        return Beam(tokens = self.tokens+[token], 
                   log_probs = self.log_probs+[log_prob],
                   sec_hidden = sec_hidden,
                   decoder_hidden = decoder_hidden,
                   encoder_outputs = encoder_outputs)
    
    @property
    def latest_token(self):
        return self.tokens[-1]
    
    @property
    def avg_log_prob(self):
        return sum(self.log_probs)/len(self.tokens)

class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, sec_encoder, decoder):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sec_encoder = sec_encoder
    
    def sort_beams(self,beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def forward(self, input_seq, input_length, sec_seq, sec_length, max_length):
        # Forward input and section through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)#(L,1,H) (layer*direc,1,H)
        sec_outputs, sec_hidden = self.sec_encoder(sec_seq, sec_length) #(secL,1,H) (layer*direc,1,H)
        
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:config.decoder_n_layers] 
        beams = [Beam(tokens=[config.SOS_token],
                     log_probs=[0.0],
                     sec_hidden=sec_hidden,
                     decoder_hidden=decoder_hidden,
                     encoder_outputs=encoder_outputs) for _ in range(config.beam_size)]
        
        all_sec_hidden = []
        all_encoder_outputs = []
        for h in beams:
                all_sec_hidden.append(h.sec_hidden)
                all_encoder_outputs.append(h.encoder_outputs)
            
        sec_hidden_stack = torch.cat(all_sec_hidden,dim=1) # (enclayer*direc,Beam,H)
        encoder_outputs_stack = torch.cat(all_encoder_outputs,dim=1) # (secL,Beam,H)
        
        results = []
        steps = 0
        while steps < max_length and len(results) < config.beam_size:
            latest_tokens = [[h.latest_token for h in beams]]
            latest_tokens = torch.tensor(latest_tokens,device=config.device, dtype=torch.long) # (1,Beam)
            
            all_decoder_hidden = []
            
            for h in beams:
                all_decoder_hidden.append(h.decoder_hidden)
            
            decoder_hidden_stack = torch.cat(all_decoder_hidden,dim=1)# (declayer,Beam,H)
            
            probs, dec_hiddens = self.decoder(latest_tokens,sec_hidden_stack[:,:len(beams)],decoder_hidden_stack,encoder_outputs_stack[:,:len(beams)])#(Beam,Vocab) (layer,Beam,H)

            log_probs = torch.log(probs)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size)

            all_beams = []
            num_old_beams = 1 if steps == 0 else len(beams)

            for i in range(num_old_beams):
                h = beams[i]
                decoder_hidden = dec_hiddens[:,i].unsqueeze(1)
                sec_hidden = sec_hidden_stack[:,i].unsqueeze(1)
                encoder_outputs = encoder_outputs_stack[:,i].unsqueeze(1)
                

                for j in range(config.beam_size):
                    new_beam = h.extend(token=topk_ids[i,j].item(),
                        log_prob=topk_log_probs[i,j].item(),
                        sec_hidden=sec_hidden,
                        decoder_hidden=decoder_hidden,
                        encoder_outputs=encoder_outputs
                        )
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == config.EOS_token:
                    results.append(h)
                else:
                    beams.append(h)

                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break 

            steps += 1

        if len(results) == 0:
            results = beams

        best_beam= self.sort_beams(results)[0]
        return best_beam.tokens[1:],best_beam.avg_log_prob