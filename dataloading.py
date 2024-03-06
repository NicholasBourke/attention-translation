from io import open
import numpy as np
import random

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler



PAD_token = 0
SOS_token = 1
EOS_token = 2


def prepare_dataset(load_path, reverse=False):
    """Create list of sentence pairs for .txt file at load_path"""
    print("Reading lines...")
    lines = open(load_path, encoding='utf-8').read().strip().split("\n")
    print("Creating sentence pairs...")
    pairs = [line.split("\t") for line in lines]
    if reverse:
        pairs = [list(reversed(pair)) for pair in pairs]
    print(f"Data subset contains {len(pairs)} sentence pairs.")
    return pairs

        
def sentence_to_indices(language, sentence):
    """Returns list of integer indices for words in sentence"""
    indices = []
    for word in sentence.split(" "):
        index = language.word2index[word]
        indices.append(index)
    indices.append(EOS_token)
    return indices


def training_dataloader(in_lang, out_lang, pairs, max_length, batch_size, valid_split=None):
    """Create training dataloader and validation dataset"""

    random.shuffle(pairs)
    n = len(pairs)

    if valid_split is not None:
        n_valid = int(np.floor(valid_split * n))
        valid_pairs = pairs[:n_valid]
        # initialize with zeros ie. PAD tokens
        valid_input_ids = np.zeros((n_valid, max_length), dtype=np.int32)
        valid_target_ids = np.zeros((n_valid, max_length), dtype=np.int32)

        print("Preparing validation dataset")
        for idx, pair in enumerate(valid_pairs):
            input, target = pair
            input_ids = sentence_to_indices(in_lang, input).append(EOS_token)
            target_ids = sentence_to_indices(out_lang, target).append(EOS_token)
            input_ids.append(EOS_token)
            target_ids.append(EOS_token)
            valid_input_ids[idx, :len(input_ids)] = input_ids
            valid_target_ids[idx, :len(target_ids)] = target_ids
        valid_data = (torch.tensor(valid_input_ids), torch.tensor(valid_target_ids))

        n_train = n - n_valid
        train_pairs = pairs[n_valid:]
    else:
        n_train = n
        train_pairs = pairs

    # initialize with zeros ie. PAD tokens
    train_input_ids = np.zeros((n_train, max_length), dtype=np.int32)
    train_target_ids = np.zeros((n_train, max_length), dtype=np.int32)

    print("Preparing training dataset")
    for idx, pair in enumerate(train_pairs):
        input, target = pair
        input_ids = sentence_to_indices(in_lang, input)
        target_ids = sentence_to_indices(out_lang, target)
        train_input_ids[idx, :len(input_ids)] = input_ids
        train_target_ids[idx, :len(target_ids)] = target_ids
    train_data = TensorDataset(torch.LongTensor(train_input_ids), torch.LongTensor(train_target_ids))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    if valid_split is not None:
        return train_dataloader, valid_data
    else:
        return train_dataloader


def evaluation_dataloader(in_lang, out_lang, pairs, max_length, batch_size):
    """Create evaluation dataloader"""

    random.shuffle(pairs)
    n = len(pairs)

    # initialize with zeros ie. PAD tokens
    eval_input_ids = np.zeros((n, max_length), dtype=np.int32)
    eval_target_ids = np.zeros((n, max_length), dtype=np.int32)

    print("Preparing evaluation dataset")
    for idx, pair in enumerate(pairs):
        input, target = pair
        input_ids = sentence_to_indices(in_lang, input)
        target_ids = sentence_to_indices(out_lang, target)
        eval_input_ids[idx, :len(input_ids)] = input_ids
        eval_target_ids[idx, :len(target_ids)] = target_ids
    eval_data = TensorDataset(torch.tensor(eval_input_ids), torch.tensor(eval_target_ids))
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    return eval_dataloader


