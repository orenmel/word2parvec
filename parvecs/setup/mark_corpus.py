'''
Input: Tokenized text corpus
Output: Text corpus with special words marked
'''

import sys

from parvecs.common.vocab import read_vocab
from parvecs.common.vocab import RARE_WORD_TOKEN
from parvecs.common.vocab import NUMERIC_TOKEN
from parvecs.common.util import is_numeric

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print >> sys.stderr, "Usage: %s <input-vocab> < corpus.txt"  % (sys.argv[0])
        sys.exit(1)


    vocab = read_vocab(sys.argv[1])    
    print >> sys.stderr, "Read vocab of size: " + str(len(vocab))
    
    i = 0
    for line in sys.stdin:
        in_words = line.split()
        out_words = []
        for word in in_words:
#            if is_numeric(word):
#                outword = NUMERIC_TOKEN
            outword = word if word in vocab else RARE_WORD_TOKEN
            out_words.append(outword)
        if len(out_words)>0:
            sys.stdout.write(' '.join(out_words) + '\n')
        i += 1
        if i % 1000000 == 0:
            print >> sys.stderr, 'Wrote ' + str(i) + ' lines'       

    
 