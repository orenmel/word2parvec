'''
word2parvec application
converts words in contexts to paraphrase vectors representations
'''
import sys
import time
import argparse
import numpy

from parvecs.inference.parvec_inferrer import ParvecInferrer
from parvecs.inference.parvec_util import parvec_lemmatize
from parvecs.common.util import vec_to_str
from parvecs.common.context_instance import read_context


def run_app(args, inferrer):
    '''
    Runs the application
    :param args: all app arguments
    :param inferrer: the parvec inferrer that is to be used
    :returns: None
    '''
    
    testfile = open(args.testfile, 'r')
    resultsfile = open(args.resultsfile, 'w')
    
    lines = 0
    last_target_key = None
    while True:
        
        try:
            context_instance, subvec = read_context(testfile, args.subvec_maxlen)
        except EOFError:
            break
        
        lines += 1
        if (args.debug == True):
            resultsfile.write("\nTest context:\n")
            resultsfile.write("=====================\n")
            
        resultsfile.write("INSTANCE\t" + context_instance.decorate_context()+'\n')
                                 
        # Assuming testfile is sorted according to target key - clear container memory every time we move to a new key target word
        if context_instance.target_key != last_target_key:
            inferrer.clear()
            last_target_key = context_instance.target_key
                
        result_vec = inferrer.infer_parvec(subvec, context_instance, resultsfile)
        
        max_vec_len = args.debugtop if args.debug == True else args.parvec_maxlen
        if (args.debug == True):
            resultsfile.write("Paraphrase vector\n")
            resultsfile.write("***************\n")
        resultsfile.write("PARVEC\t" + vec_to_str(result_vec, max_vec_len)+"\n")
        
        if (args.lemmatize == True):
            result_vec_lemmatized = parvec_lemmatize(result_vec, context_instance.partofspeech)
            if (args.debug == True):
                resultsfile.write("Lemmatized paraphrase vector\n")
                resultsfile.write("***************\n")
            resultsfile.write("PARLEMVEC\t" + vec_to_str(result_vec_lemmatized, max_vec_len)+"\n")
        
        if lines % 100 == 0:
            print "Read %d lines" % lines                      
        
    print "Read %d word instances in total" % lines 
    print "Net processing time for computing the paraphrase vectors per each word instance: %f msec" % inferrer.msec_per_word()          
    testfile.close()
    resultsfile.close()
    
    
def run(args):
    '''
    Initialize inferrer and run app
    :param args:
    '''
    
    print "Initializing"
    print time.asctime(time.localtime(time.time()))
    
    inferrer = ParvecInferrer(args)
    print "Running"
    print time.asctime(time.localtime(time.time()))
    
    run_app(args, inferrer)
    print "Finished"
    print time.asctime(time.localtime(time.time()))


    
if __name__ == '__main__':
  
    parser = argparse.ArgumentParser(description='Parvec App')

    parser.add_argument('--debug',action='store_true',dest='debug')
    parser.add_argument('-debugtop', action="store", dest="debugtop", type=int, default=10, help="Top number of vector entries to print in debug mode.")
    
    parser.add_argument('-contexts_dir', action="store", dest="contexts_dir", default=None)
    parser.add_argument('-vocabfile', action="store", dest="vocabfile", default=None)
    parser.add_argument('-testfile', action="store", dest="testfile", default=None)
    parser.add_argument('-resultsfile', action="store", dest="resultsfile", default=None)
    
    parser.add_argument('--lemmatize', action="store_true", dest="lemmatize", default=False, help="Lemmatize output paraphrase vectors.")    
    parser.add_argument('-parvec_maxlen', action="store", dest="parvec_maxlen", type=int, default=100, help="Max num of paraphrases in each output parvec.")
    parser.add_argument('-subvec_maxlen', action="store", dest="subvec_maxlen", type=int, default=None, help="Max num of substitutes read per subvec.")    
    parser.add_argument('-top', action="store", dest="top", type=int, default=0, help="Num of top most similar contexts to consider for each given context. 0 means all context.")
    parser.add_argument('-toppercent', action="store", dest="top_percent", type=float, default=0.0, help="Percent of top contexts to consider. Param 'top' is considered as min number to consider in any case. 0 means all contexts.")
    parser.add_argument('-weightsfactor',action='store',dest='weightsfactor', type=float, default=1.0, help="Context similarity weights power factor.")
    parser.add_argument('--excluderef',action='store_true',dest='excluderef', default=False, help="Exclude reference (given) context from context averaging.")

    parser.add_argument('--pmi',action='store_true',dest='pmi', default=False, help="Convert conditional probability substitute weights in input files to pmi (or spmmi) weights).")    
    parser.add_argument('-pmioffset',action='store',dest='pmioffset', type=float, default=0.0, help='pmi=pmi-offset')
    parser.add_argument('-pmithreshold',action='store',dest='pmithreshold', type=float, default=0.0, help='pmi=0 if pmi<=threshold')

    parser.add_argument('-bow',action='store',dest='bow_size', default=-1, type=int, help="Context bag-of-words window size used for computing context sim. -1 means bow is not used, 0 means entire sentence.")
    parser.add_argument('-bowinter',action='store',dest='bow_interpolate', default=0.0, type=float, help="Interpolation factor between bow and subvec context sims. 0 means only consider subvec similarity.")
    parser.add_argument('-cbow',action='store',dest='embeddingpath', default=None, help="Use continuous bow (embeddings avg) instead of bow")
    
    parser.add_argument('--tfidf',action='store_true',dest='tfidf', default=False, help="Use tfidf weighting in bow.")    
    parser.add_argument('-tfidfoffset',action='store',dest='tfidf_offset', type=float, default=0.0, help='tfidf=tfidf-offset')
    parser.add_argument('-tfidfthreshold',action='store',dest='tfidf_threshold', type=float, default=0.0, help='tfidf=0 if tfidf<=threshold')
    parser.add_argument('--nostopwords',action='store_false',dest='use_stopwords', default=True)
    
    
    
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
    
