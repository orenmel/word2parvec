'''
Input: Tokenized text corpus
Output: Vocabulary counts
'''

import sys
import string
from operator import itemgetter
from parvecs.common.vocab import VOCAB_TOTAL_COUNT


if __name__ == '__main__':
    
    if len(sys.argv)<2:
        print >> sys.stderr, "Usage: %s <min-count> < corpus.txt" % sys.argv[0]
        sys.exit(1)
        
    min_count = int(sys.argv[1])
    vocab = {}
    i = 0
    for line in sys.stdin:
        words = line.split()
        for word in words:
            if (word not in vocab):
                vocab[word] = 1
            else:
                vocab[word] +=1
            i += 1
            if i % 10000000 == 0:
                print >> sys.stderr, 'Read ' + str(i) + ' words'          
    vocab[VOCAB_TOTAL_COUNT] = i    
    sorted_vocab = sorted(vocab.iteritems(), key=itemgetter(1), reverse=True)   
    for word, count in sorted_vocab:
        if count < min_count:
            break;
        print '\t'.join([word, str(count)])    