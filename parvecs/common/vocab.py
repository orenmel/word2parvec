VOCAB_TOTAL_COUNT = "<VOCAB_TOTAL_COUNT>"
RARE_WORD_TOKEN = "<RW>"
NUMERIC_TOKEN = "<NUM>"
STOPWORD_TOP_THRESHOLD = 256

import sys

def read_vocab(vocab_filename):
    vocab = {}
    with open(vocab_filename,'r') as f:
        for line in f:
            tokens = line.split('\t')
            word = tokens[0].strip()
            count = int(tokens[1])
            vocab[word] = count
    return vocab

def vocab_total_size(vocab):
    return vocab[VOCAB_TOTAL_COUNT]

def load_vocabulary_w2i(vocab_filename):
    with open(vocab_filename) as f:
        vocab = [line.split('\t')[0].strip() for line in f if len(line) > 0]
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab

def load_vocabulary_counts(path):
    stop_words = set()
    counts = {}
    with open(path) as f:
        i = 0
        for line in f:
            if len(line) > 0:
                tokens = line.split('\t') 
                word = tokens[0].strip() 
                count = int(tokens[1].strip())
                counts[word] = count
                i += 1 
                if (i <= STOPWORD_TOP_THRESHOLD):
                    stop_words.add(word)
    total_size = counts[VOCAB_TOTAL_COUNT]                
    return counts, total_size, stop_words