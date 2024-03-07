import random
import pickle
from datetime import date

from dataloading import prepare_dataset, training_dataloader, evaluation_dataloader
from layers import EncoderRNN, AttnDecoderRNN
from training import train, evaluate, translate

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


today = date.today().strftime("%y%m%d")
PAD_token = 0
SOS_token = 1
EOS_token = 2

# load french and english Language class instances 
with open("data/french.pkl", "rb") as f:
    french = pickle.load(f)
f.close()

with open("data/english.pkl", "rb") as e:
    english = pickle.load(e)
e.close()


K_x = french.n_words    # source vocabulary size
K_y = english.n_words   # target vocabulary size

M = 620                 # word embedding dimensionality
N = 1000                # hidden layer size
P = 1000                # alignment model hidden layer size
L = 500                 # maxout hidden layer size in deep output
B = 80                  # batch size

eta = 0.001             # learning rate
valid_split = 0.05      # proportion of dataset saved for validation


data_subsets = {
        1: {"max_length": 7, "path": "T=3-7"},
        2: {"max_length": 9, "path": "T=8-9"},
        3: {"max_length": 12, "path": "T=10-12"},
        4: {"max_length": 16, "path": "T=13-16"},
        5: {"max_length": 56, "path": "T=17-56"}
    }

encoder = EncoderRNN(K_x, M, N)
enc_optimizer = optim.Adam(encoder.parameters(), lr=eta)

loss_fn = nn.NLLLoss()


# Function to train a single data subset
def train_subset(subset, data_path, load_paths, save_path):
    
    T = data_subsets[subset]["max_length"]
    path = data_subsets[subset]["path"]
    print(F"----- TRAINING SUBSET {subset} ({path}) -----")

    data_path = data_path + f"_{path}.txt"
    pairs = prepare_dataset(data_path, reverse=True)

    dataloader = training_dataloader(french, english, pairs, T, B)

    decoder = AttnDecoderRNN(K_y, T, M, N, P, L)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=eta)

    if load_paths is not None:
        encoder.load_state_dict(torch.load(load_paths[0][0]))
        decoder.load_state_dict(torch.load(load_paths[0][1]))

    save_path += f"[{path}]"
    load_paths = train(dataloader, encoder, decoder, enc_optimizer, dec_optimizer, loss_fn, save_path)

    return load_paths, save_path


# Set to "training", "evaluation", or "translation" mode
mode = 



if mode == "training":
    save_path = "statesaves/240217/"

    for subset in data_subsets:
        path = data_subsets[subset]["path"]
        data_path = f"data/eng-fra_clean_training90_{path}.txt"
        pairs = prepare_dataset(data_path, reverse=True)

        dataloader = training_dataloader(french, english, pairs, T, B)

        T = data_subsets[subset]["max_length"]

        decoder = AttnDecoderRNN(K_y, T, M, N, P, L)
        dec_optimizer = optim.Adam(decoder.parameters(), lr=eta)

        if subset > 1:
            encoder.load_state_dict(torch.load(load_paths[0][0]))
            decoder.load_state_dict(torch.load(load_paths[0][1]))

        save_path += f"[{path}]"
        load_paths = train(dataloader, encoder, decoder, enc_optimizer, dec_optimizer, loss_fn, save_path)



if mode == "evaluation":
    T = 36

    data_path = "data/eng-fra_clean_evaluation10.txt"
    pairs = prepare_dataset(data_path, reverse=True)

    dataloader = evaluation_dataloader(french, english, pairs, T, 200)

    encoder = EncoderRNN(K_x, M, N)
    decoder = AttnDecoderRNN(K_y, T, M, N, P, L)
    enc_state_path =    # set to save path of last training run
    dec_state_path =    # set to save path of last training run
    encoder.load_state_dict(torch.load(enc_state_path))
    decoder.load_state_dict(torch.load(dec_state_path))

    accuracy = evaluate(dataloader, encoder, decoder)
    print(f"ACCURACY = {accuracy:.2f}")



if mode == "translation":
    T = 36

    data_path = f"data/eng-fra_clean_evaluation10.txt"
    pairs = prepare_dataset(data_path, reverse=True)

    sentences = random.choices(pairs, k=10)

    encoder = EncoderRNN(K_x, M, N)
    decoder = AttnDecoderRNN(K_y, T, M, N, P, L)

    enc_state_path =    # set to save path of last training run
    dec_state_path =    # set to save path of last training run
    encoder.load_state_dict(torch.load(enc_state_path))
    decoder.load_state_dict(torch.load(dec_state_path))

    for pair in sentences:
        translation = translate(pair, french, english, encoder, decoder)
        print(f"FRENCH:  {pair[0]}")
        print(f"ENGLISH: {pair[1]}")
        print(f"MODEL:   {translation}\n")




