'''
Input: context subvecs filename
Output: new directory named filename.DIR. In this directory a subvecs file per each target individually)
'''

import sys
import os

SUBVEC_DIR_SUFFIX = ".DIR"

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print "Usage: %s input_subvec_file" % sys.argv[0]
        sys.exit(1)
        
    input_subvec_filename = sys.argv[1]
    
    subvec_dirname = input_subvec_filename + SUBVEC_DIR_SUFFIX
    os.mkdir(subvec_dirname)
    
    input_subvec_file = open(input_subvec_filename, 'r')
    
    output_files = {}
    
    while True:
        line1 = input_subvec_file.readline()
        line2 = input_subvec_file.readline()        
        if not line1 or not line2:
            break;
        
        target = line1[:line1.find('\t')]
        if target not in output_files:
            output_files[target] = open(subvec_dirname + "/" + target, 'w') 
        
        output_files[target].write(line1)
        output_files[target].write(line2)
    
    input_subvec_file.close()
    for output_file in output_files.itervalues():
        output_file.close()