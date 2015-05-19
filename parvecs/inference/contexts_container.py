'''
A container of context collections of different target words
'''

from parvecs.inference.context_collection import ContextCollection
from parvecs.common.util import count_file_lines


class ContextsContainer():


    def __init__(self, args, w2i, i2w, w2counts, sum_word_counts, stopwords, embeddings):
        
        self.args = args
        self.container = {}
        self.w2i = w2i
        self.i2w = i2w
        self.w2counts = w2counts
        self.sum_word_counts = sum_word_counts
        self.stopwords = stopwords
        self.embeddings = embeddings
        
        
    def get_target_contexts(self, target):        
        '''
        :param target: target word
        :returns: context collection for target word
        '''
        try:
            if target not in self.container:
                self.load_target_contexts(target)
            return self.container[target]
        except IOError as e:
            return None
                            
            
    def load_target_contexts(self, target):
        '''
        load into memory the contexts of target
        :param target: target word
        '''
        target_filename = self.args.contexts_dir+"/"+target        
        collection_size = count_file_lines(target_filename)/2 # subvec every two lines
        target_subfile = open(target_filename, 'r')
        self.container[target] = ContextCollection(self.args, self.i2w, self.w2i, collection_size, self.w2counts, self.sum_word_counts, self.stopwords, self.embeddings)
        self.container[target].load_contexts(target_subfile)
        if len(self.container[target].contexts) != collection_size:
            raise EOFError('context collection size mismatch in target %s. collection_size %d len(contexts) %d' % (target, collection_size, len(self.container[target].contexts)))
        self.container[target].tocsr()
        target_subfile.close()
        
    
    def clear(self):
        '''
        clear memory of container
        '''
        self.container = {}
        
        