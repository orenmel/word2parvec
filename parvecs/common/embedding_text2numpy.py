'''
Converts the word2vec embeddings format to a numpy-friendly format
'''
import numpy as np
import sys


def readVectors(path):
    vectors = {}
    with open(path) as input_f:
        for line in input_f.readlines():
            tokens = line.strip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
    return vectors

inpath = sys.argv[1]
outpath = sys.argv[2]

matrix = readVectors(inpath)

vocab = list(matrix.keys())
vocab.sort()
with open(outpath+'.vocab', 'w') as output_f:
    for word in vocab:
        print >>output_f, word,

new_matrix = np.zeros(shape=(len(vocab), len(matrix[vocab[0]])), dtype=np.float32)
for i, word in enumerate(vocab):
    new_matrix[i, :] = matrix[word]

np.save(outpath+'.npy', new_matrix)