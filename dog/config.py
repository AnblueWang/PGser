import torch

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw token

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
beam_size = 5
use_cuda = True

data_path = '../Conversations/seq+att/train'
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
MAX_HISTORY_LENGTH = 100 #Maximum history length 
MAX_LENGTH = 25  # Maximum sentence length to consider
MIN_COUNT = 2    # Minimum word count threshold for trimming
hidden_size = 300
embedding_size = 100
encoder_n_layers = 2
decoder_n_layers = 1
dropout = 0.3
batch_size = 64
# Configure training/optimization
clip = 20.0
teacher_forcing_ratio = 0.7
learning_rate = 0.0001
decoder_learning_ratio = 3.0
n_iteration = 30000
print_every = 100
save_every = 500

# Set checkpoint to load from; set to None if starting from scratch
# loadFilename = '../Conversations/seq+att/cb_model/2-1_300/25000_checkpoint.tar'
checkpoint_iter = 4000