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
from data_utils import batchGenerator
import config

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(config.device)
    return loss, nTotal.item()

def train(input_variable, lengths, section_variable, sec_lengths, sec_idx, target_variable, mask, max_target_len, encoder, sec_encoder, decoder, embedding, encoder_optimizer, sec_encoder_optimizer, decoder_optimizer):

    # Zero gradients
    encoder_optimizer.zero_grad()
    sec_encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(config.device)
    lengths = lengths.to(config.device)
    section_variable = section_variable.to(config.device)
    sec_lengths = sec_lengths.to(config.device)
    sec_idx = sec_idx.to(config.device)
    target_variable = target_variable.to(config.device)
    mask = mask.to(config.device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths) # (L,B,H)  (layer*direc,B,H)
    try:
        # Forward pass through encoder
        sec_outputs, sec_hidden = sec_encoder(section_variable, sec_lengths) # (secL,B,H)  (layer*direc,B,H)
        sec_hidden = sec_hidden.index_select(1,sec_idx) #调整回按utter长度排序的batch内顺序
    except Exception as e:
        print(section_variable,section_variable.shape)
        print(sec_lengths,sec_lengths.shape)
        print(input_variable,input_variable.shape)
        print(lengths,lengths.shape)
        raise e

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[config.SOS_token for _ in range(config.batch_size)]]) # (1,B)
    decoder_input = decoder_input.to(config.device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers] #（layer,B,H)

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < config.teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, sec_hidden, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, sec_hidden, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(config.batch_size)]])
            decoder_input = decoder_input.to(config.device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.clip)
    _ = torch.nn.utils.clip_grad_norm_(sec_encoder.parameters(), config.clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.clip)

    # Adjust model weights
    encoder_optimizer.step()
    sec_encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(voc, pairs, wiki_strings, encoder, sec_encoder, decoder, encoder_optimizer, sec_encoder_optimizer, decoder_optimizer, embedding,save_dir):

    # Load batches for each iteration
    training_batch_generator = batchGenerator(voc,pairs,wiki_strings)

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
#     if config.loadFilename:
#         start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, config.n_iteration + 1):
        training_batch = next(training_batch_generator)
        # Extract fields from batch
        doc_input, doc_lengths, input_variable, lengths, target_variable, mask, max_target_len = training_batch
        
        #将doc按长度降序排列，并保存让其恢复原样的idx2
        doc_lengths,idx1 = torch.sort(doc_lengths,descending=True)
        doc_input = doc_input.index_select(1,idx1)
        _,idx2 = torch.sort(idx1)
        # Run a training iteration with batch
        loss = train(input_variable, lengths, doc_input, doc_lengths, idx2, target_variable, mask, max_target_len, encoder,sec_encoder,decoder, embedding, encoder_optimizer,sec_encoder_optimizer, decoder_optimizer)
        print_loss += loss

        # Print progress
        if iteration % config.print_every == 0:
            print_loss_avg = print_loss / config.print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / config.n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % config.save_every == 0):
            directory = os.path.join(save_dir, config.model_name, '{}-{}_{}'.format(config.encoder_n_layers, config.decoder_n_layers, config.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'sec_en':sec_encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'sec_en_opt': sec_encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))