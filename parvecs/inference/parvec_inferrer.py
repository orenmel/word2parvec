'''
ParvecInferrer generates a paraphrase vector for a target word in a given context

'''
import heapq
import time

from parvecs.common.vocab import load_vocabulary_w2i
from parvecs.common.vocab import load_vocabulary_counts
from parvecs.common.context_instance import get_pmi_weights
from parvecs.common.context_instance import remove_out_of_vocab
from parvecs.common.util import wf2ws
from parvecs.common.util import vec_to_str
from parvecs.common.util import TimeRecorder
from parvecs.common.embedding import Embedding
from parvecs.inference.contexts_container import ContextsContainer


class ParvecInferrer():


    def __init__(self, args):
        
        self.args = args
        self.bow_interpolate = self.args.bow_interpolate
        self.w2i, self.i2w = load_vocabulary_w2i(args.vocabfile)    
        self.w2counts, self.sum_word_counts, self.stopwords = load_vocabulary_counts(args.vocabfile)
        if args.use_stopwords == False:
            self.stopwords = {}
        print "Vocab size: " + str(len(self.w2i))
             
        if args.embeddingpath != None:
            embeddings = Embedding(args.embeddingpath)
            print "Read embeddings from " + args.embeddingpath
        else:
            embeddings = None
            
        self.context_container = ContextsContainer(args, self.w2i, self.i2w, self.w2counts, self.sum_word_counts, self.stopwords, embeddings)
        self.time_recorder = TimeRecorder()
        
                    
    def clear(self):
        '''
        Clears the contexts cache
        '''
        self.context_container.clear()
        
        
    def infer_parvec(self, subvec, context_instance, tfo):
        '''
        generate the paraphrase vector
        :param orig_subvec: subvec of instance
        :param context_instance: context instance
        :param tfo: output file
        :returns: parvec
        '''
    
        subvec = self.__preprocess_subvec(subvec, context_instance, tfo)
        
        if (self.args.debug == True):
            tfo.write("\nUsing weightsfactor %s\n" % ('{0:1.1f}'.format(self.args.weightsfactor)))
        
        target_contexts = self.context_container.get_target_contexts(context_instance.target)
        
        if target_contexts is not None:
            
            start1 = time.time()
            subvec_matrix = target_contexts.reference_context(subvec, context_instance, self.bow_interpolate)
            end1 = time.time()
                    
            if (self.args.debug == True) and (self.bow_interpolate > 0):
                tfo.write("\nUsed BOW similarity. bow_interpolate = %f\n\n" % self.bow_interpolate)  
            
            max_len = self.args.debugtop if self.args.debug == True else len(subvec) 
            trimmed_sorted_subvec = heapq.nlargest(max_len, subvec, key=lambda t: t[1])  
            tfo.write("SUBVEC\t" + '\t'.join([' '.join([word, wf2ws(weight)]) for (word, weight) in trimmed_sorted_subvec])+'\n')        
            
            start2 = time.time()
            result_vec, contexts_num = target_contexts.avg_contexts(subvec_matrix, self.args.top, self.args.top_percent, self.args.parvec_maxlen, self.args.excluderef, self.args.weightsfactor)
            end2 = time.time()
            
            deltatime = (end1-start1) + (end2-start2)
            self.time_recorder.iteration_time(deltatime)
            
            if (self.args.debug == True):
                tfo.write("\nDeltatime: %f msec\n" % (deltatime*1000))
                tfo.write("\nTop similar contexts:\n")
                tfo.write("**************************\n")
                tfo.write(target_contexts.to_str(min(self.args.debugtop,contexts_num) , self.args.debugtop)+"\n\n")
    
            if (self.args.debug == True):
                if (result_vec is not None):
                    tfo.write("Avg of top " + str(contexts_num) + " contexts: " + vec_to_str(result_vec, self.args.debugtop) + '\n')
                else:
                    tfo.write("Avg of top " + str(contexts_num) + " contexts: None\n")                     
                tfo.write("*****************************************\n\n") 
        else:
            if (self.args.debug == True):
                tfo.write("\nNo subvecs found for target [%s], using only reference subvec.\n" % context_instance.target)
            tfo.write("SUBVEC\t" + '\t'.join([' '.join([word, wf2ws(weight)]) for (word, weight) in subvec])+'\n')
            result_vec = subvec                  
                
        return result_vec
            

    def msec_per_word(self):
        '''
        returns: mean net processing time per parvec generation
        '''
        return self.time_recorder.msec_per_iteration()
    

    def __preprocess_subvec(self, subvec, context_instance, tfo):        
        if (self.args.pmi == True):
            subvec = get_pmi_weights(subvec, self.w2counts, self.sum_word_counts, self.args.pmioffset, self.args.pmithreshold)
        else:
            subvec = remove_out_of_vocab(subvec, self.w2counts)                   
        return sorted(subvec, reverse=True, key=lambda x: x[1])
    
 