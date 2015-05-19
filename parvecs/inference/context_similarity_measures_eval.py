'''
This application is used to evaluate context similarity measures using pseudowords.
'''
import sys
import argparse
from random import Random
import numpy

from parvecs.common.vocab import load_vocabulary_w2i
from parvecs.common.vocab import load_vocabulary_counts
from parvecs.inference.context_collection import ContextCollection
from parvecs.common.embedding import Embedding
from parvecs.common.util import count_file_lines


def read_pseudo_words(pseudos_filename):
    '''
    Reads pseudo words from file
    :param pseudos_filename:
    :returns: mapping of each pseudoword to a list of pseudosense
    '''   
    words = []
    f = open(pseudos_filename, 'r')
    for line in f:
        word = line[:line.find('\t')]
        pseudos = line.split()[1:]        
        words.append((word, pseudos))   
    f.close()
    
    return words

def evaluate_word(word, collection, seeded_random, results_file):
    '''
    Evaluate context similarity measures on a given pseudoword experiment
    :param word: the pseudoword used to perform the experiment
    :param collection: the contexts of the pseudoword
    :param seeded_random:
    :param results_file:
    :returns: evaluation results
    '''
    
    m_precision_at_1 = 0.0 # mean precision@1
    m_top_precision = 0.0 # mean precision@top
    m_avg_precision = 0.0 # mean average precision
    for i in xrange(0,args.sample_num):
        precision_at_1, top_precision, avg_precision, debug_str = collection.evaluate_context_similarity(seeded_random, args.random_sim)
        m_precision_at_1 += precision_at_1
        m_top_precision += top_precision
        m_avg_precision += avg_precision
        if args.debug:
            results_file.write("\n" + debug_str + "\n")
            results_file.write("%d: p@1: %f \t p@top: %f \t avg_p: %f\n" % (i, precision_at_1, top_precision, avg_precision))
            
    m_precision_at_1  /= args.sample_num
    m_top_precision  /= args.sample_num
    m_avg_precision  /= args.sample_num
            
    print "Mean over all samples for word [%s]: m_p@1: %f \t m_p@top :%f \t m_avg_p: %f\n" % (word, m_precision_at_1, m_top_precision, m_avg_precision)
    if args.debug:
        results_file.write("\nMean over all samples for word [%s]: m_p@1: %f \t m_p@top: %f \t m_avg_p: %f\n\n" % (word, m_precision_at_1, m_top_precision, m_avg_precision))
    else:
        results_file.write("%s\t%f\t%f\t%f\n" % (word, m_precision_at_1, m_top_precision, m_avg_precision))
    results_file.flush()
    return m_precision_at_1, m_top_precision, m_avg_precision


def add_pseudo_word_to_vocab(i2w, w2i, w2counts, word, pseudo_senses):
    '''
    Add pseudo word to vocabulary
    :param i2w:
    :param w2i:
    :param w2counts:
    :param word:
    :param pseudo_senses:
    :returns: the label of the pseudoword in the vocabulary
    '''
    label = word+'='+'+'.join(pseudo_senses)
    i2w.append(label)
    w2i[label] = len(i2w)-1
    count = 0
    for word in pseudo_senses:
        count += w2counts[word]
    w2counts[label] = count 
    return label   

def run(args):
    '''
    Run application
    :param args:
    '''
    
    w2i, i2w = load_vocabulary_w2i(args.vocabfile)    
    w2counts, sum_word_counts, stopwords = load_vocabulary_counts(args.vocabfile)
    print "Vocab size: " + str(len(w2i))
    
    if args.embeddingpath != None:
        embeddings = Embedding(args.embeddingpath)
        print "Read embeddings from " + args.embeddingpath
    else:
        embeddings = None

    words = read_pseudo_words(args.pseudos_file)
    
    results_file = open(args.resultsfile,'w')
    
    mm_precision_at_1 = 0.0 # mean mean precision@1
    mm_top_precision = 0.0 # mean mean precision@top
    mm_avg_precision = 0.0 # mean mean average precision
    seeded_random = Random()
    for word in words:           
        
        word_name = word[0]
        
#        word_seed = word_name+' '+' '.join(word[1])
#        the 'star' is used for backward compatibility with previous experiments
        word_name_star = word_name+'.*'
        word_star = [pseudo+'.*' for pseudo in word[1]]
        word_seed = word_name_star+' '+' '.join(word_star)        
        
        seeded_random.seed(word_seed) # we want the same random numbers when repeating experiments with different params etc.    
        senses = word[1]
        
        collection_size = 0
        for target in senses:
            target_filename = args.contexts_dir+"/"+target        
            collection_size += count_file_lines(target_filename)/2 # subvec every two lines
        
        pseudos_label = add_pseudo_word_to_vocab(i2w, w2i, w2counts, word_name, senses)    
        collection = ContextCollection(args, i2w, w2i, collection_size, w2counts, sum_word_counts, stopwords, embeddings)
         
        if args.debug:
                results_file.write("Reading word for word_name [%s]\n" % word_name)                  
        for target in senses:
            target_filename = args.contexts_dir+"/"+target                
            target_subfile = open(target_filename, 'r')
            lines_num = collection.load_contexts(target_subfile, set(senses), pseudos_label, tocsr_flag=False)
            if args.debug:
                results_file.write("Read %d contexts for pseudo [%s]\n" % (lines_num, target))        
            target_subfile.close()
        collection.tocsr()    
        m_precision_at_1, m_top_precision, m_avg_precision = evaluate_word(word_name, collection, seeded_random, results_file) 
            
        mm_precision_at_1 += m_precision_at_1
        mm_top_precision += m_top_precision
        mm_avg_precision += m_avg_precision
            
    mm_precision_at_1 /= len(words)
    mm_top_precision /= len(words)
    mm_avg_precision /= len(words)
    
    results_file.write("TOTAL\t%f\t%f\t%f\n" % (mm_precision_at_1, mm_top_precision, mm_avg_precision))
    
    if args.debug:
        results_file.write("#WORDS\t%d\n" % len(words))
        results_file.write("MM_P1\t%f\n" % (mm_precision_at_1))
        results_file.write("MM_PTOP\t%f\n" % (mm_top_precision))
        results_file.write("MM_AVG\t%f\n" % (mm_avg_precision))
        
    results_file.close()





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Context similarity measures app')
    
    parser.add_argument('--debug',action='store_true',dest='debug', default=False)

    parser.add_argument('-samplenum', action="store", dest="sample_num", type=int, default=None, help="number of samples from each pseudowords collection")
    parser.add_argument('-pseudosfile', action="store", dest="pseudos_file", default=None)
    parser.add_argument('-contexts_dir', action="store", dest="contexts_dir", default=None)
    parser.add_argument('-vocabfile', action="store", dest="vocabfile")
    parser.add_argument('-resultsfile', action="store", dest="resultsfile")
    parser.add_argument('-embeddingpath', action="store", dest="embeddingpath", default=None, help="prefix to files containing word embeddings")
    

    parser.add_argument('-top', action="store", dest="top", type=int, help="num of top contexts to consider")
    parser.add_argument('-toppercent', action="store", dest="top_percent", type=float, default=0.0, help="percent of top contexts to consider. When using this, top num is considered as min number to consider")
    parser.add_argument('-subvec_maxlen', action="store", dest="subvec_maxlen", type=int, default=None, help="max num of substitutes read per subvec")
    
    parser.add_argument('--randomsim',action='store_true',dest='random_sim', default=False, help='similarity measure returns zero for all context pairs')
    parser.add_argument('--pmi',action='store_true',dest='pmi', default=False)    
    parser.add_argument('-pmioffset',action='store',dest='pmioffset', type=float, default=0.0, help='pmi=pmi-offset')
    parser.add_argument('-pmithreshold',action='store',dest='pmithreshold', type=float, default=0.0, help='pmi=0 if pmi<=threshold')
    
    parser.add_argument('--tfidf',action='store_true',dest='tfidf', default=False)    
    parser.add_argument('-tfidfoffset',action='store',dest='tfidf_offset', type=float, default=0.0, help='tfidf=tfidf-offset')
    parser.add_argument('-tfidfthreshold',action='store',dest='tfidf_threshold', type=float, default=0.0, help='tfidf=0 if tfidf<=threshold')
    
    
    parser.add_argument('-weightsfactor',action='store',dest='weightsfactor', type=float, default=1.0, help="context similarity measure power factor")
    parser.add_argument('-bow',action='store',dest='bow_size', default=-1, type=int, help="context bag-of-words window size for context cosine sim. -1 means bow not used, 0 means entire sentence")
    parser.add_argument('-bowinter',action='store',dest='bow_interpolate', default=0, type=float, help="interpolation factor between bow and subvec sims. 0 means no bow, -1 means doing backoff instead of interpolation.")
    parser.add_argument('-cbow',action='store',dest='embeddingpath', default=None, help="continuous bow (embeddings avg)")
    
    if len(sys.argv)==1:
        print parser.print_help(sys.stdout)
    else:
    
        args = parser.parse_args(sys.argv[1:])
        
        config_file_name = args.resultsfile + ".CONFIG"
        cf = open(config_file_name, 'w')
        cf.write(' '.join(sys.argv)+'\n')
        cf.close()
        
        numpy.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
        
        run(args)
        
    
        