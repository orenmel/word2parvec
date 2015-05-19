'''
Converts substitute weights from conditional probabilities to pmi (or sppmi)
'''
import sys
from operator import itemgetter

from parvecs.common.vocab import read_vocab
from parvecs.common.vocab import vocab_total_size
from parvecs.common.context_instance import ContextInstance
from parvecs.common.context_instance import read_context
from parvecs.common.context_instance import get_pmi_weights


def write_subvec(output, subvec):
    for word, weight in subvec:
        output.write(word + " " + '{0:1.8f}'.format(weight) + "\t")
    output.write("\n")
        
        
if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: %s <vocab-file> <pmi-shift> [normalize]  <input >output\n" % sys.argv[0])
        sys.exit(1)
        
    vocab = read_vocab(sys.argv[1])
    total_size = vocab_total_size(vocab)
    pmi_shift = float(sys.argv[2])
    normalize = False
    if len(sys.argv) > 3 and sys.argv[3] == 'normalize':
        normalize = True
 
    lines = 0    
    try:
        while True: 
            context_inst, subvec = read_context(sys.stdin)
            subvec_pmi = get_pmi_weights(subvec, vocab, total_size, pmi_shift, 0.0, normalize)
            sorted_subvec_pmi = sorted(subvec_pmi, key=itemgetter(1), reverse=True)
            sys.stdout.write(context_inst.line+'\n')
            write_subvec(sys.stdout,sorted_subvec_pmi)                                         
            lines += 1
            if lines % 10000 == 0:
                sys.stderr.write("Read %d subvecs\n" % (lines))
    except EOFError:            
        sys.stderr.write("Finished loading %d context lines\n" % lines)
    
        
    