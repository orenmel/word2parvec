'''
Input: context subvecs file
Output: only the context subvecs for which the original target word that was observed in this context appears in the subvec
'''


import sys
from parvecs.common.context_instance import ContextInstance

if __name__ == '__main__':
    
    if len(sys.argv)<3:
        print "Usage: %s <input-subvec-file> <output-subvec-file> <output-targetfreqs-file>" % sys.argv[0]
        sys.exit(1)
    
    input_sub_file = open(sys.argv[1], 'r')
    output_sub_file = open(sys.argv[2], 'w')
    output_targetfreqs_file = open(sys.argv[3], 'w')
    target_freqs = {}
    
    while True:
        context_line = input_sub_file.readline()
        subs_line = input_sub_file.readline()
        if not context_line or not subs_line:
            break
        
        context_inst = ContextInstance(context_line)
        
        if context_inst.target != context_inst.target_key:
            sys.stderr.write("Skipping bad context: " + context_line)
            continue
        
        substitute_words = subs_line.split()[::2] 
        
        if context_inst.target in substitute_words:
            output_sub_file.write(context_line)
            output_sub_file.write(subs_line)
            if context_inst.target in target_freqs:
                target_freqs[context_inst.target] = target_freqs[context_inst.target]+1
            else:
                target_freqs[context_inst.target] = 1  

    for word, freq in sorted(target_freqs.iteritems(), key=lambda x: x[1], reverse=True):
        output_targetfreqs_file.write("%s\t%d\n" % (word, freq))
        
    input_sub_file.close()
    output_sub_file.close()
    output_targetfreqs_file.close()

 