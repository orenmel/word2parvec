import re
import string
import heapq

def asciize(line):
    return filter(lambda x: x in string.printable, line)

def is_printable(s):
    return all(c in string.printable for c in s)

# very crude implementation
num_re = re.compile('^[\+\/\:\-,\.\d]*\d[\+\/\:\-,\.\d]*$')
def is_numeric(word_str):
    return num_re.match(word_str) != None

def wf2ws(weight):
        return '{0:1.5f}'.format(weight)
    
def vec_to_str(subvec, max_n):
    sub_list_sorted = heapq.nlargest(max_n, subvec, key=lambda x: x[1])
    sub_strs = [' '.join([word, wf2ws(weight)]) for word, weight in sub_list_sorted]
    return '\t'.join(sub_strs)    
    
def count_file_lines(filename):
    f = open(filename, 'r')
    lines_num = sum(1 for line in f)
    f.close()
    return lines_num

class TimeRecorder(object):
    
    def __init__(self):
        self.time = 0.0
        self.iterations = 0
     
     
    def iteration_time(self, seconds):
        self.time += seconds
        self.iterations += 1
                   
     # processing time in msec
    def msec_per_iteration(self):
        return 1000*self.time/self.iterations if self.iterations > 0 else 0.0

