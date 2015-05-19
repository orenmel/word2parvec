'''
Represents the given context of a target word instance 
'''

CONTEXT_TEXT_BEGIN_INDEX = 3

import math

class ContextInstance(object):
 
    def __init__(self, line):
        '''
        Constructor
        
        Example line:
        bright.a        1       13      during the siege , george robertson had appointed shuja-ul-mulk , who was a bright boy
        '''
        self.line = line
        tokens1 = line.split("\t")
        self.target_key = tokens1[0]
        self.target_id = tokens1[1]
        self.target_ind = int(tokens1[2])
        self.target = tokens1[3].split()[self.target_ind]
        pos_delimiter_ind = self.target_key.rfind('.')
        if pos_delimiter_ind > 0 and pos_delimiter_ind == len(self.target_key)-2:
            self.partofspeech = self.target_key[pos_delimiter_ind+1:]
        else:
            self.partofspeech = None
   
   
    def get_context_tokens(self):
        '''
        :returns: a list of the text tokens
        '''
        all_tokens = self.line.split()
        return all_tokens[CONTEXT_TEXT_BEGIN_INDEX:]

    
    
    def get_neighbors(self, window_size):
        '''
        Get the neighbors of a target word
        :param window_size: neighbors window size
        :returns: a list of neighbors
        '''
        tokens = self.line.split()[3:]
        
        if (window_size > 0):                                    
            start_pos = max(self.target_ind-window_size, 0)
            end_pos = min(self.target_ind+window_size+1, len(tokens))
        else:
            start_pos = 0
            end_pos = len(tokens)
            
        neighbors = tokens[start_pos:self.target_ind] + tokens[self.target_ind+1:end_pos]
        return neighbors 
   
    
    def decorate_context(self):
        '''
        :returns the context text line with target word highlighted
        '''
        tokens = self.line.split('\t')
        words = tokens[CONTEXT_TEXT_BEGIN_INDEX].split()
        words[self.target_ind] = '__'+words[self.target_ind]+'__'
        tokens[CONTEXT_TEXT_BEGIN_INDEX] = ' '.join(words)
        return '\t'.join(tokens)
    

def read_context(subfile, maxlen=None):
    '''
    Reads a context and substitute vector from file
    :param subfile:
    :param maxlen:
    :returns context instance, subvec
    '''
    context_line = subfile.readline()
    subvecs_line = subfile.readline()
    if not context_line or not subvecs_line:
        raise EOFError

    context_inst = ContextInstance(context_line.strip())
    subvecs_line = subvecs_line.strip()
    subvec = [__extract_word_weight(pair) for pair in subvecs_line.split("\t")[:maxlen]] if len(subvecs_line) > 0 else []
    
    return context_inst, subvec



def get_pmi_weights(subvec, w2counts, sum_counts, shift, threshold, normalize=False):
    '''
    Converts a subvec with conditional probability weights to pmi (or sppmi) weights
    Also performs the functionality of remove_out_of_vocab
    :param subvec:
    :param w2counts:
    :param sum_counts:
    :param shift:
    :param threshold:
    :param normalize:
    :returns: subvec with pmi weights
    '''
    subvec_pmi = []
    norm = 0
    for word, prob in subvec:
        if prob != 0.0 and word in w2counts:
            pmi = math.log(prob * sum_counts / w2counts[word])-shift
            if pmi>threshold:
                subvec_pmi.append((word, pmi))
                norm += pmi**2
            
    if normalize:
        norm = norm**0.5
        for i in xrange(0,len(subvec_pmi)):
            subvec_pmi[i] = (subvec_pmi[i][0], subvec_pmi[i][1] / norm)       
            
    return subvec_pmi

def remove_out_of_vocab(subvec, w2counts):
    '''
    Removes entries from subvec that are out of the vocabulary
    :param subvec:
    :param w2counts:
    :returns: subvec in vocab
    '''
    subvec_vocab = []
    for word, prob in subvec:
        if prob != 0.0 and word in w2counts:
            subvec_vocab.append((word, prob))
    return subvec_vocab


def normalize_subvec(subvec):
    '''
    normalizes subvec weights in L2
    :param subvec:
    :returns: normalized subvec
    '''
    norm = 0.0
    for word, weight in subvec:
        norm += weight**2
    norm = norm**0.5
    for i in xrange(0,len(subvec)):
        subvec[i] = (subvec[i][0], subvec[i][1] / norm)  
        

def __extract_word_weight(pair):
    tokens = pair.split(' ')
    return tokens[0], float(tokens[1])   

  