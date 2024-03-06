import random
import pickle
from datetime import date

from dataloading import prepare_dataset, training_dataloader, evaluation_dataloader
from layers import EncoderRNN, AttnDecoderRNN
from training import train, evaluate, translate

import torch.nn as nn
from torch import optim
import torch.nn.functional as F



# Load French and English Language class instances 
with open("data/french.pkl", "rb") as f:
    french = pickle.load(f)
f.close()

with open("data/english.pkl", "rb") as e:
    english = pickle.load(e)
e.close()


# Hyperparameters
batch_size = 80
eta = 0.001             # learning rate
valid_split = 0.05      # proportion of dataset saved for validation


# Model Dimensions
K_x = french.n_words    # source vocabulary size
K_y = english.n_words   # target vocabulary size
M = 620                 # word embedding dimensionality
N = 1000                # hidden layer size
P = 1000                # alignment model hidden layer size
L = 500                 # maxout hidden layer size in deep output

# Set max sentence length (source and target). Has a significant
# affect on training time (see README.txt for discussion).
T = 56         


encoder = EncoderRNN(K_x, M, N)
decoder = AttnDecoderRNN(K_y, T, M, N, P, L)


# TRAINING
train_path = "data/eng-fra_clean_training90.txt"
train_pairs = prepare_dataset(train_path, reverse=True)
train_dataloader = training_dataloader(french, english, train_pairs, T, batch_size)

enc_optimizer = optim.Adam(encoder.parameters(), lr=eta)
dec_optimizer = optim.Adam(decoder.parameters(), lr=eta)
loss_fn = nn.NLLLoss()

today = date.today().strftime("%y%m%d")
save_path = f"data/{today}"
epoch_save_paths = train(train_dataloader, encoder, decoder, enc_optimizer, dec_optimizer, loss_fn, save_path, n_epochs=3)


# EVALUATION
eval_path = "data/eng-fra_clean_evaluation10.txt"
eval_pairs = prepare_dataset(eval_path, reverse=True)
eval_dataloader = evaluation_dataloader(french, english, eval_pairs, T, 200)

accuracy = evaluate(eval_dataloader, encoder, decoder)
print(f"ACCURACY = {accuracy:.2f}")

sentences = random.choices(eval_pairs, k=10)
for pair in sentences:
    translation = translate(pair, french, english, encoder, decoder)
    print(f"FRENCH:  {pair[0]}")
    print(f"ENGLISH: {pair[1]}")
    print(f"MODEL:   {translation}\n")


