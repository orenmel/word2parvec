'''
Generates pseudowords using Wordnet
'''
import sys
import random
import re
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from parvecs.common.vocab import load_vocabulary_counts

DEBUG = False
MIN_FREQ = 1000 # words representing pseudo-senses need to have this min frequency in the learning corpus

word_re = re.compile('^[a-z][a-z]+$')
pos_list = set([wn.NOUN, wn.VERB,wn.ADJ, wn.ADV])


def sample_target_word(vocab_counts, stop_words, min_freq):
    '''
    Create a single pseudoword (randomly sampled from vocab)
    :param vocab_counts:
    :param stop_words:
    :param min_freq: minimum required corpus frequency of word
    '''
    
    accum_counts = []
    n = 0
    accum_counts.append((None, 0))
    for word, count in vocab_counts.iteritems():
        if count >= min_freq and word not in stop_words:
            n += count
            accum_counts.append((word, n))
    max_count = n
    
    while True:
        rnd = random.randint(0, max_count) 
        for k in xrange(1,len(accum_counts)):
            sampled_word = accum_counts[k][0]
            if len(wn.synsets(sampled_word))>1 and word_re.match(sampled_word) != None:
                if rnd < accum_counts[k][1] and rnd >= accum_counts[k-1][1]:
                    return sampled_word
                
    print "Failed to sample target word"
    sys.exit(1)
    

if __name__ == '__main__':
    
    if len(sys.argv) < 5:
        print "usage: %s <vocab-file> <words-num> <words2senses-file> <senses-file> [<min-freq>]"  % (sys.argv[0])
        sys.exit(1)
        
    stemmer = PorterStemmer()
    
    vocab_file = sys.argv[1]
    words_num = int(sys.argv[2])
    words2senses_file = open(sys.argv[3], 'w')
    senses_file = open(sys.argv[4], 'w')
    if len(sys.argv) > 5:
        min_freq = int(sys.argv[5])
    else:
        min_freq = 1000
    
    vocab_counts, ignore, stop_words = load_vocabulary_counts(vocab_file)
    
    words = set()
    all_words = set()
    
    while len(words) < words_num:
        while True:
            word = sample_target_word(vocab_counts, stop_words, min_freq)
            if word not in words:
                break; 
        word_synsets = wn.synsets(word)
        if DEBUG: print "Word: [%s] Number of senses: %s" % (word, str(len(word_synsets)))
        
        senses = set()
        for word_synset in word_synsets:
            if DEBUG: print "\tsynset: %s" % word_synset
            pos = word_synset.pos()
            if pos in pos_list:
                sense = None
                smallest_sense_num_found = sys.maxint
                for lemma in word_synset.lemmas():           
                    if DEBUG: print  "\t " + lemma.name(), len(wn.synsets(lemma.name()))                   
                    if (stemmer.stem(lemma.name()) != stemmer.stem(word)) and \
                    WordNetLemmatizer().lemmatize(lemma.name(), pos) != WordNetLemmatizer().lemmatize(word, pos) and \
                    (lemma.name().islower()) and (lemma.name() in vocab_counts) and (vocab_counts[lemma.name()]>=min_freq) and \
                    (lemma.name() not in stop_words) and (word_re.match(lemma.name()) != None) and \
                    (len(wn.synsets(lemma.name())) < smallest_sense_num_found): # we look for the lemma with least number of senses, i.e. hopefully least ambiguous                         
                        sense = lemma.name()
                        smallest_sense_num_found = len(wn.synsets(lemma.name()))                
                if sense != None:
                    if DEBUG: print "\tChosen sense word: %s %d\n" % (sense, smallest_sense_num_found)                
                    senses.add(sense)
                else:
                    if DEBUG: print "\tDidn't find any suitable sense word. Skipping.\n"
        if len(senses) > 1:
                all_words.update(senses)            
                sys.stdout.write(word + ':\t' + ' '.join(senses)+'\n')
                words2senses_file.write(word + '\t' + ' '.join(senses)+'\n')
                words.add(word)    
    
    for pword in all_words:
        senses_file.write(pword+"\n")
            
    words2senses_file.close()
    senses_file.close()
        