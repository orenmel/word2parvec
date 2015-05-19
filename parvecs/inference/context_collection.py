'''
A collection of contexts represented as subvecs and/or bags-of-words
Used to:
- sort contexts according to their similarity to a reference context
- perform a (weighted) average of contexts representations

'''
from parvecs.common.context_instance import read_context
from parvecs.common.util import wf2ws
from parvecs.common.context_instance import get_pmi_weights
from parvecs.common.context_instance import remove_out_of_vocab
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse import SparseEfficiencyWarning
import numpy as np
import heapq
import math

import warnings
warnings.simplefilter('error',SparseEfficiencyWarning)



class ContextCollection():

    def __init__(self, args, i2w, w2i, subvecs_num, w2counts, sum_word_counts, stopwords, embeddings):
        
        self.args = args        
        self.w2i = w2i
        self.i2w = i2w
        self.w2counts = w2counts
        self.sum_word_counts = sum_word_counts
        self.stopwords = stopwords
        
        self.contexts = []
        self.sim_scores = None # points either to self.subvecs_sim_scores or to self.bow_sim_scores  
        
        initial_sim_score = 1.0 if subvecs_num==0 else 1.0/subvecs_num 
        
        self.embeddings = embeddings # when this is not None the bow representation is dense (todo: refactor this code)
        self.bow_size = args.bow_size
        if (self.bow_size >= 0):
            if (self.embeddings != None):
                bow_dimensionality = self.embeddings.dimension()
                self.bow_matrix = np.zeros((subvecs_num, bow_dimensionality), dtype=np.float32) # estimate sim of contexts based on their BOW rep
                self.bow_L2_norms = None # we always keep them normalized
                self.bow_sim_scores = dok_matrix([[initial_sim_score]*subvecs_num]).tocsr().transpose()
            else:
                bow_dimensionality = len(w2i)
                self.bow_matrix = dok_matrix((subvecs_num, bow_dimensionality), dtype=np.float32) # estimate sim of contexts based on their BOW rep
                self.bow_L2_norms = dok_matrix((subvecs_num, 1), dtype=np.float32)
                self.bow_sim_scores = dok_matrix([[initial_sim_score]*subvecs_num]).tocsr().transpose()
                
        self.subs_matrix = dok_matrix((subvecs_num, len(w2i)), dtype=np.float32) #used for sim weights calculation, also for sub average only if no dual matrix
        self.subvecs_L2_norms = dok_matrix((subvecs_num, 1), dtype=np.float32)
        self.subvecs_sim_scores = dok_matrix([[initial_sim_score]*subvecs_num]).tocsr().transpose()
        
        self.target_counts = {} 
        
        
    def load_contexts(self, contexts_file, pseudos=None, pseudos_label=None, tocsr_flag=True):
        '''
        loads contexts for this collection
        :param contexts_file:
        :param pseudos: set of pseudo-sense words (used only for pseudo-word experiments)
        :param pseudos_label: pseudo-word label (used only for pseudo-word experiments)
        :param tocsr_flag: should be False if intending to load more contexts into this collection
        :returns: number of contexts read
        '''
        
        print "Loading contexts for file %s" % (contexts_file)
        lines = 0
        try:
            while True: 
                context_instance, subvec = read_context(contexts_file, self.args.subvec_maxlen)
                if pseudos != None and pseudos_label != None:
                    subvec = self.__update_pseudos(subvec, pseudos, pseudos_label)
                
                if self.args.pmi == True:
                    subvec = get_pmi_weights(subvec, self.w2counts, self.sum_word_counts, self.args.pmioffset, self.args.pmithreshold)
                else:
                    subvec = remove_out_of_vocab(subvec, self.w2counts)                  
                self.__append_subvec(subvec, context_instance)
                
                lines += 1
                if lines % 10000 == 0:
                    print "Read %d context lines" % (lines)
        except EOFError:
            print "Finished loading %d context lines from file %s" % (lines, contexts_file)
        if tocsr_flag == True:
            self.tocsr()     
        return lines
    

    def tocsr(self):
        '''
        Converts collection to an arithmetically-efficient format
        :returns: None
        '''        
        self.subs_matrix = self.subs_matrix.tocsr()
        self.subvecs_L2_norms = self.subvecs_L2_norms.tocsr()       
        if self.bow_size>=0:
            if isinstance(self.bow_matrix, dok_matrix):
                self.bow_matrix = self.bow_matrix.tocsr()
                self.bow_L2_norms = self.bow_L2_norms.tocsr()
        
    def reference_context(self, subvec, context, bow_interpolate):
        '''
        Weighs contexts in this collection according to similarity to the given reference context
        :param subvec: subvec representation of given context
        :param context: given context
        :param bow_interpolate: interpolation factor (between bow and subvec simiarity)
        :returns: subvec as a numpy matrix
        '''        
        subvec_matrix = dok_matrix((len(self.w2i),1), dtype=np.float32)       
        for word, weight in subvec:
            subvec_matrix[self.w2i[word],0] = weight    
        subvec_matrix = subvec_matrix.tocsr()       
        
        return self.__reference_context_imp(subvec_matrix, context, bow_interpolate)
    

    def avg_contexts(self, ref_subvec, top, top_percent, top_inferences_number, exclude_ref, weights_factor):
        '''
        Performs a weighted average of
        :param ref_subvec: given subvec as a numpy matrix
        :param top:
        :param top_percent:
        :param top_inferences_number:
        :param exclude_ref:
        :param weights_factor:
        :returns: parvec, number of contexts averaged
        '''
        
        if len(self.contexts) == 0:
            return None, 0
        
        ref_weight = 1 if exclude_ref == False else 0
               
        if (top > len(self.contexts) + ref_weight):
            top = len(self.contexts) + ref_weight
            
        if (top > 0 or top_percent > 0):
            top_contexts_weights = self.sim_scores.todok()
            final_top = top-ref_weight # -1 to leave 1 for the ref_subvec
            num_top_percent = int(math.ceil(top_percent * (len(self.contexts)+ref_weight)))-ref_weight
            final_top = max(final_top, num_top_percent) 
            
            cw_sorted  = heapq.nlargest(final_top, top_contexts_weights.iteritems(), key=lambda x: x[1])
            top_contexts_weights = dok_matrix((len(self.contexts),1), dtype=np.float32)
            
            for (k,j), weight in cw_sorted:
                top_contexts_weights[k,j] = weight**weights_factor

            top_contexts_weights = top_contexts_weights.tocsr()
            contexts_num = len(cw_sorted)
                
        else:            
            contexts_num = len(self.contexts)
            if weights_factor == 0.0:
                top_contexts_weights = dok_matrix([[1.0]*contexts_num]).tocsr().transpose()
            else:
                top_contexts_weights = self.sim_scores.copy()
                top_contexts_weights.data **= weights_factor
            
        sum_weights = top_contexts_weights.sum() + ref_weight #weight +1 reserved for ref_subvec
        top_contexts_weights.data /= sum_weights


        weighted_subs_matrix = self.subs_matrix.multiply(top_contexts_weights)  #NOT SUPPORTED IN SCIPY 0.7        
        avg_subvec = weighted_subs_matrix.sum(axis=0)
        
        if (exclude_ref == False) and (ref_subvec != None):
            ref_subvec.data *= 1.0/sum_weights
            avg_subvec = avg_subvec + ref_weight * ref_subvec.transpose()
        
        result_vec = self.__vec_to_sorted_list(avg_subvec, top_inferences_number)  
        return result_vec, contexts_num        


    def evaluate_context_similarity(self, seeded_random, random_similarity):
        '''
        Performs a context similarity measure evaluation on a single 'query' context instance
        todo: move this functionality out of this class
        :param seeded_random:
        :param debug_top_inferences_per_context:
        :param random_similarity:
        :returns: precision results
        '''
        
        random_context_ind = seeded_random.randint(0, len(self.contexts)-1)
        sample_context = self.contexts[random_context_ind]
        sample_target = self.contexts[random_context_ind].target
        sample_subvec = self.subs_matrix[random_context_ind,:].transpose()
        
        all_size = len(self.contexts)-1  # -1 because we used one context as query
        all_real_pos = self.target_counts[sample_target]-1
        
        if (self.args.top > 0 or self.args.top_percent > 0):
            top_contexts = self.args.top
            num_top_percent = int(math.ceil(self.args.top_percent * all_size))
            top_contexts = max(top_contexts, num_top_percent) 
        else:
            top_contexts = all_size
        
        bow_interpolate = self.args.bow_interpolate

        if random_similarity:
            self.sim_scores = csr_matrix([]) 
        else:
            self.__reference_context_imp(sample_subvec, sample_context, bow_interpolate)
        
        contexts_weights_sorted = sorted(self.sim_scores.todok().iteritems(), key=lambda x: x[1], reverse=True)
        output_items = []
        true_p = 0
        all_p = 0
        precision_at_1 = None
        top_precision = None
        avg_precision = 0.0
        
        # going over all the contexts that got a non-zero score
        for i in xrange(0,len(contexts_weights_sorted)):
            (j,k), context_weight = contexts_weights_sorted[i]
            retrieved_target = self.contexts[j].target
            
            if j != random_context_ind: # skipping the sampled context in calculation
                all_p += 1
                if retrieved_target == sample_target: # true positive 
                    true_p += 1
                    avg_precision += float(true_p) / all_p  
                if all_p == 1:
                    precision_at_1 = float(true_p) / all_p
                if all_p == top_contexts:
                    top_precision = float(true_p) / all_p
                                                   
            if self.args.debug:
                subvec = self.subs_matrix_for_sim_weights[j, :].todok()
                sub_list_sorted  = heapq.nlargest(self.args.debugtop, subvec.iteritems(), key=lambda x: x[1])
                sub_strs = [' '.join([self.i2w[ii], wf2ws(weight)]) for (kk,ii), weight in sub_list_sorted]
                prefix = "QRY" if j == random_context_ind else "RET"
                output_items.append((prefix, context_weight, self.contexts[j].decorate_context() +'\n' +'\t' + '\t'.join(sub_strs)))                
            
        # for all the contexts that got zero score (were not retrieved at all) we assume that the real positives were retrieved uniformly (like random)
        false_n = all_real_pos - true_p
        if (false_n > 0):
            all_n = all_size - all_p
            real_negs_per_one_real_pos = (float(all_n)/false_n)-1
            
            all_p += real_negs_per_one_real_pos/2
            
            while all_p < all_size:                        
                if (top_precision == None)  and (all_p >= top_contexts):                
                    top_precision = float(true_p) / top_contexts
                all_p += 1
                true_p += 1
                avg_precision += float(true_p) / all_p            
                all_p += real_negs_per_one_real_pos
                if self.args.debug:
                    output_items.append(("UNF", 0.0, "dummy positive"))
        
        if (top_precision == None):                
                top_precision = float(true_p) / top_contexts
                
        if (precision_at_1 == None):                
                precision_at_1 = float(all_real_pos) / all_size
            
        assert(true_p == all_real_pos)
        
        avg_precision /= max(1,all_real_pos)
        
        output_lines = ['\t'.join([prefix, wf2ws(context_weight), text]) for prefix, context_weight, text in output_items]  
        return precision_at_1, top_precision, avg_precision,'\n'.join(output_lines)


    def __append_subvec(self, subvec, context_instance):
       
        j = len(self.contexts)
        self.contexts.append(context_instance)        
        
        if context_instance.target in self.target_counts:
            self.target_counts[context_instance.target] += 1
        else:
            self.target_counts[context_instance.target] = 1
        
        if len(subvec) > 0:
            L2 = 0.0
            for word, weight in subvec:
                L2 += weight**2
            if L2 == 0:
                L2 = 1
            self.subvecs_L2_norms[j,0] = 1.0/(L2**0.5)
             
            for word, weight in subvec:
                if (weight != 0):
                    self.subs_matrix[j, self.w2i[word]] = weight 
        else:
            self.subvecs_L2_norms[j,0] = 1.0 # dummy NORM
            
            
        if self.bow_size >= 0: # using the bow_matrix for sim between contexts
            
            text_matrix, found_word = self.__context_text_to_vec(context_instance)
            
            if (self.embeddings == None):
                text_matrix = text_matrix.transpose()
                
                for (zero, word_ind), value in text_matrix.iteritems():
                    self.bow_matrix[j, word_ind] = value
                
                if found_word == True:
                    L2 = 0
                    for val in text_matrix.itervalues():
                        L2 += val**2       
                    self.bow_L2_norms[j,0] = 1.0 / (L2**0.5)
                else:             
                    self.bow_L2_norms[j,0] = 1.0 # dummy NORM
            else:
                self.bow_matrix[j, :] = text_matrix

        
    def __reference_context_imp(self, subvec_matrix, context, bow_interpolate):
                      
        if bow_interpolate == 1:
            self.bow_sim_scores = self.__reference_context_bow(context)
            self.sim_scores = self.bow_sim_scores            
        elif bow_interpolate == 0:
            self.subvecs_sim_scores = self.__reference_context_subvec(subvec_matrix)
            self.sim_scores = self.subvecs_sim_scores
        else:
            try: 
                self.bow_sim_scores = self.__reference_context_bow(context)
                self.bow_sim_scores.data = self.bow_sim_scores.data**bow_interpolate
            except Exception as e:
                print e
                print context
                raise e 
            self.subvecs_sim_scores = self.__reference_context_subvec(subvec_matrix)
            self.subvecs_sim_scores.data = self.subvecs_sim_scores.data**(1-bow_interpolate)
            self.sim_scores = self.subvecs_sim_scores.multiply(self.bow_sim_scores)
                               
        return subvec_matrix

   
    def __reference_context_bow(self, context):
        
        refvec_matrix, found_word = self.__context_text_to_vec(context)
        sims = self.__compute_sim_scores(refvec_matrix, self.bow_matrix, self.bow_L2_norms, self.embeddings != None)       
        return sims
        
    
    def __reference_context_subvec(self, refvec_matrix):        
        sims = self.__compute_sim_scores(refvec_matrix, self.subs_matrix, self.subvecs_L2_norms, False)        
        return sims
    
    
    
    def __compute_sim_scores(self, refvec_matrix, allvecs_matrix, L2_norms, is_embeddings):
        contexts_sims = allvecs_matrix.dot(refvec_matrix)        
                
        if is_embeddings:
            contexts_sims = (contexts_sims + 1) / 2 # map cosine to [0,1]
            contexts_sims = np.reshape(contexts_sims, (len(contexts_sims), 1))
            contexts_sims = csr_matrix(contexts_sims.tolist())     
        if L2_norms != None:
            contexts_sims = contexts_sims.multiply(L2_norms)           
            refvec_dp = refvec_matrix.transpose().dot(refvec_matrix)
            refvec_L2_norm = refvec_dp.data.max()**0.5 if len(refvec_dp.data) > 0 else 1.0
            contexts_sims.data /= refvec_L2_norm # weights -1 <= cosine <= 1, but in practice greater than zero because all weights >= 0

        return contexts_sims
    
    def __context_text_to_vec(self, context_instance):
        found_word = False        
        
        if self.embeddings != None:
            dimensionality = self.embeddings.dimension()
            weight_dtype = np.float32
            w2ind = self.w2i
            text_matrix = np.zeros((dimensionality,), dtype=weight_dtype)           
        else:
            dimensionality = len(self.w2i)
            weight_dtype = np.float32 if self.args.tfidf else np.int8
            w2ind = self.w2i
            text_matrix = dok_matrix((dimensionality,1), dtype=weight_dtype)
        
        context_text_tokens = context_instance.get_context_tokens()
        target_pos = context_instance.target_ind
        
        if (self.bow_size > 0):                                    
            start_pos = max(target_pos-self.bow_size, 0)
            end_pos = min(target_pos+self.bow_size+1, len(context_text_tokens))
            context_text_tokens = context_text_tokens[start_pos:end_pos]
            target_pos = target_pos-start_pos
                       
        stopwords = self.stopwords
        context_text_inds_left = [w2ind[word] for word in context_text_tokens[:target_pos] if word not in stopwords and word in w2ind]    
        context_text_inds_right = [w2ind[word] for word in context_text_tokens[target_pos+1:] if word not in stopwords and word in w2ind] if (target_pos+1) < len(context_text_tokens) else []
                             
        all_words_inds = context_text_inds_left+context_text_inds_right
        total_weights = 0.0
        for word_ind in all_words_inds:
            w = self.i2w[word_ind]
            if self.args.tfidf:                
                wcount = self.w2counts[w]
                log_idf = math.log(float(self.sum_word_counts)/wcount)
                log_idf -= self.args.tfidf_offset
                if (log_idf <= self.args.tfidf_threshold):
                    log_idf = 0.0
                weight = log_idf
            else:
                weight = 1
            
            if weight !=0:
                found_word = True
                if (self.embeddings != None):
                    if w in self.embeddings:                    
                        wordvec = self.embeddings.represent(w).transpose()
                        text_matrix = text_matrix + (wordvec * weight)
                    else:
                        weight = 0.0
                else:
                    text_matrix[word_ind,0] += weight
                total_weights += weight
          
        # embeddings representations are always normalized
        if (self.embeddings != None):
            if total_weights != 0:
                text_matrix /= total_weights
            norm = np.sqrt(np.sum(text_matrix*text_matrix))
            if norm != 0:
                text_matrix /= norm

        return text_matrix, found_word
    
    
    def __vec_to_sorted_list(self, subvec, max_n):
        sub_list = np.array(subvec)[0].tolist()
        n = min(max_n, subvec.nonzero()[0].shape[1]) if max_n > 0 else subvec.nonzero()[0].shape[1] 
        sub_list_sorted  = heapq.nlargest(n, enumerate(sub_list), key=lambda x: x[1])
        sub_list = [(self.i2w[i], weight) for i, weight in sub_list_sorted]
        return sub_list
    
               
    def to_str(self, top_contexts, top_inferences_per_context):
        
        contexts_weights_sorted = heapq.nlargest(top_contexts, self.sim_scores.todok().iteritems(), key=lambda x: x[1])
        output_items = []
        for (j,k), context_weight in contexts_weights_sorted:
            subvec = self.subs_matrix[j, :].todok()
            sub_list_sorted  = heapq.nlargest(top_inferences_per_context, subvec.iteritems(), key=lambda x: x[1])
            sub_strs = [' '.join([self.i2w[i], wf2ws(weight)]) for (k,i), weight in sub_list_sorted]
            output_items.append((context_weight, self.contexts[j].decorate_context() +'\n' + '\t'.join(sub_strs)))
         
        output_lines = ['\t'.join([wf2ws(context_weight), text]) for context_weight, text in output_items]  
        return '\n'.join(output_lines)
        
    
    def __update_pseudos(self, subvec, pseudos, pseudos_label):
        updated_subvec = []
        pseudos_weight = 0.0
        for word, weight in subvec:
            if word in pseudos:
                pseudos_weight += weight
            else:
                updated_subvec.append((word, weight))
        if pseudos_weight > 0.0:
            updated_subvec.append((pseudos_label, pseudos_weight))
        
        return sorted(updated_subvec, key=lambda x: x[1], reverse=True)
        
    
       
