'''
utilities manipulating paraphrase vectors
'''
import re
import heapq
from nltk.stem.wordnet import WordNetLemmatizer

lemmatized_word_re = re.compile('^[a-zA-Z\-]+$')

def parvec_lemmatize(parvec, target_pos):    
    '''
    lemmatizes a paraphrase vector
    :param parvec: input parvec
    :param target_pos: part-of-speech used for lemmatization
    :returns lemmatized parvec
    '''
    
    lemmas = {}
    if parvec is not None:
        for word, weight in parvec:
            if lemmatized_word_re.match(word) != None: # filter out non-words
                lemma = WordNetLemmatizer().lemmatize(word, target_pos)
                if lemma in lemmas:
                    weight = max(weight, lemmas[lemma])
                lemmas[lemma] = weight
    parlemvec = sorted(lemmas.iteritems(), key=lambda x: x[1], reverse=True) 
    return parlemvec