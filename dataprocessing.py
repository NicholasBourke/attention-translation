from __future__ import unicode_literals, print_function, division
import io
import unicodedata
import re
import numpy as np
import random
import pickle


PAD_token = 0
SOS_token = 1
EOS_token = 2


class Language:
    """Language class that contains dictionaries to convert words to indices
    (and vice-versa) and determine vocabulary size and word counts"""

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        # adds all words in sentence to language
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        # adds word to language and updates count, assigns index if new
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



def unicode_to_ascii(s):
    """Turns a Unicode string to plain ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    """Lowercase, trim, and remove non-letter characters"""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z'!?]+", r" ", s)
    return s.strip()

def clean_strings(load_path, save_path):
    """Loads strings from .txt file at load_path, cleans strings,
    and saves strings to .txt file at save_path"""

    lines = io.open(load_path, encoding='utf-8').read().strip().split("\n")
    pairs = [[normalize_string(s) for s in line.split("\t")] for line in lines]

    clean = io.open(save_path, "w")
    for pair in pairs:
        line = pair[0]+"\t"+pair[1]+"\n"
        clean.write(line)

    clean.close()



def train_eval_split(eval_prop, load_path, save_path):
    """Splits full dataset into training and evaluation datasets"""

    lines = io.open(load_path, encoding='utf-8').read().strip().split("\n")
    pairs = [line.split("\t") for line in lines]

    n = len(pairs)
    random.shuffle(pairs)
    n_eval = int(np.floor(eval_prop * n))
    pairs_eval = pairs[:n_eval]
    pairs_train = pairs[n_eval:]

    save_path_eval = f"{save_path}evaluation{int(eval_prop*100)}.txt"
    save_path_train = f"{save_path}training{int((1-eval_prop)*100)}.txt"

    lines_eval = io.open(save_path_eval, "w")
    for pair in pairs_eval:
        line = pair[0]+"\t"+pair[1]+"\n"
        lines_eval.write(line)
    lines_eval.close()

    lines_train = io.open(save_path_train, "w")
    for pair in pairs_train:
        line = pair[0]+"\t"+pair[1]+"\n"
        lines_train.write(line)
    lines_train.close()



def create_languages(lang1, lang2, dataset_path, reverse=False):
    """Creates language class instances and adds words"""

    lines = open(dataset_path, encoding='utf-8').read().strip().split("\n")
    pairs = [line.split("\t") for line in lines]

    if reverse:
        pairs = [list(reversed(pair)) for pair in pairs]
        in_lang = Language(lang2)
        out_lang = Language(lang1)
    else:
        in_lang = Language(lang1)
        out_lang = Language(lang2)

    print("Adding all sentences to Language classes...")
    for pair in pairs:
        in_lang.add_sentence(pair[0])
        out_lang.add_sentence(pair[1])
    print(f"{in_lang.name} language consists of {in_lang.n_words} words")
    print(f"{out_lang.name} language consists of {out_lang.n_words} words")

    return in_lang, out_lang



# Create and populate language class instances and save with pickle
dataset_path = "data/eng-fra_clean.txt"
french, english = create_languages("English", "French", dataset_path, reverse=True)

with open("data/french.pkl", "wb") as f:
    pickle.dump(french, f)
f.close()

with open("data/english.pkl", "wb") as e:
    pickle.dump(english, e)
e.close()







# Split dataset by sentence length

def filter_length(pairs, T_max):
    """Filters list of sentence pairs by sentence length"""
    short_pairs = [pair for pair in pairs if len(pair[0].split(" ")) < (T_max-2) and len(pair[1].split(" ")) < (T_max-2)]
    return short_pairs
    
def max_length(pair):
    """Determines max length of a sentence pair (incl. SOS and EOS)"""
    return max(len(pair[0].split(" ")), len(pair[1].split(" "))) + 2

def filter_max_length(pairs, a, b):
    """Filters list of sentence pairs by max sentence length range"""
    return [pair for pair in pairs if max_length(pair) >= a and max_length(pair) < b]

def count_max_length(L, load_path):
    """Counts number of pairs with max sentence length between range points in L"""
    lines = io.open(load_path, encoding='utf-8').read().strip().split("\n")
    pairs = [line.split("\t") for line in lines]

    counts = []
    for l in range(L+1):
        pairs_l = [pair for pair in pairs if max_length(pair) == l]
        counts.append(len(pairs_l))
    return counts

def length_split(breaks, path):
    """Creates datasets split by max sentence length"""
    load_path = path + ".txt"
    lines = io.open(load_path, encoding='utf-8').read().strip().split("\n")
    pairs = [line.split("\t") for line in lines]

    for i in range(len(breaks)-1):
        print(f"writing {breaks[i]} to {breaks[i+1]-1}")
        pairs_ab = filter_max_length(pairs, breaks[i], breaks[i+1])
        print(len(pairs_ab))
        save_path_ab = f"{path}_T={breaks[i]}-{breaks[i+1]-1}.txt"
        lines_ab = io.open(save_path_ab, "w")
        for pair in pairs_ab:
            line = pair[0]+"\t"+pair[1]+"\n"
            lines_ab.write(line)
        lines_ab.close()




