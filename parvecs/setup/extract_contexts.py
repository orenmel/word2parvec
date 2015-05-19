'''
Input: one-sentence-per-line tokenized text corpus and list of target words
Output: contexts of target words
'''

import sys
from operator import itemgetter
import string

MAX_WORDS_IN_LINE = 128
MAX_CHARS_IN_LINE = 1024


if __name__ == '__main__':
    
    if len(sys.argv) < 6:
        print >> sys.stderr, "Usage: %s <corpus-file> <targets-file> <max-freq> <contexts-file> <targets-freqs-file>"  % (sys.argv[0])
        sys.exit(1)
     
    corpus_file = sys.argv[1]   
    targets_file = sys.argv[2]
    max_freq = int(sys.argv[3])
    contexts_file = sys.argv[4]
    targets_freq_file = sys.argv[5]
    
    targets = {}
    with open(targets_file,'r') as tf:
        for line in tf:
            word = line.split('\t')[0].strip()
            targets[word] = 0                
    print >> sys.stderr, "Read %d targets " % (len(targets))
    
    cf = open(corpus_file,'r')
    mf = open(contexts_file , 'w')
    
    i = 0
    full_targets = 0
    for line in cf:    
        if len(line) < MAX_CHARS_IN_LINE:
            stripped_line = line.strip()
            sent_words = stripped_line.split()
            if len(sent_words) <= MAX_WORDS_IN_LINE:                        
                for ind, word in enumerate(sent_words):   
                    if (word in targets and targets[word] < max_freq):
                        mf.write('\t'.join([word, str(i), str(ind), stripped_line])+'\n')
                        targets[word] += 1
                        if targets[word] == max_freq:
                            full_targets += 1
                i += 1 
                if i % 1000000 == 0:
                    print >> sys.stderr, 'Read ' + str(i) + ' lines'
                if (full_targets == len(targets)):
                    break
                  
    cf.close()
    mf.close()
    
    with open(targets_freq_file, 'w') as tff:
        for target, freq in sorted(targets.iteritems(), key=itemgetter(1), reverse=True):
            tff.write(target + '\t' + str(freq) + '\n')