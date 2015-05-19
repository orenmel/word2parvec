'''
Created on 22.9.14

@author: user
'''
from parvecs.common.context_instance import read_context
from parvecs.common.context_instance import normalize_subvec
from parvecs.common.vocab import load_vocabulary_w2i

import sys
import os
import os.path
from operator import itemgetter
import numpy as np
from scipy.sparse.dok import dok_matrix

from sklearn.cluster import KMeans


def normalize_centroids(centroids):
    for j in xrange(0,len(centroids)):
        norm = (np.dot(centroids[j,:],centroids[j,:]))**0.5
        if norm > 0:
            centroids[j,:] /= norm


def cluster_subvec_file(w2i, cluster_prunning, K, ninit, maxiter, min_avg_cluster_size, subvec_filename, cluster_filename):
    '''
    kmeans clustering of subvecs given in an input file
    :param w2i: word2index
    :param cluster_prunning: max size of a cluster centroid
    :param K: number of clusters
    :param ninit: number of repeating tries
    :param maxiter: number of clustering iterations
    :param min_avg_cluster_size: min size of clusters (on average)
    :param subvec_filename: input filename
    :param cluster_filename: output filename
    :returns: None
    '''
    
    if os.path.exists(cluster_filename):
        print "NOTICE: cluster file %s already exists. skipping." % cluster_filename 
        return   
        
    subvec_file = open(subvec_filename, 'r')
    subvec_num = sum(1 for line in subvec_file)/2 #subvec is on every second line
    subvec_file.seek(0)
    
    minK = min(subvec_num/min_avg_cluster_size, K)
    minK = max(1, minK)
      
    cluster_file = open(cluster_filename, 'w')    
    print "Clustering subvecs in file %s. Using K=%d\n" % (cluster_filename, minK)       
        
    target = subvec_filename[subvec_filename.rfind('/')+1:]
    subs_matrix = dok_matrix((subvec_num, len(w2i)), dtype=np.float32)
    
    line = 0    
    try:
        while True: 
            context_inst, subvec = read_context(subvec_file)
            normalize_subvec(subvec)
            for word, weight in subvec:
                if (weight != 0):
                    subs_matrix[line, w2i[word]] = weight 
            line += 1
            if line % 10000 == 0:
                sys.stderr.write("Read %d subvecs\n" % (line))
    except EOFError:            
        sys.stderr.write("Finished loading %d context lines\n" % line)
        
    subs_matrix = subs_matrix.tocsr()
        
    best_centroids = None
    best_inertia = None
    
    for init_iter in xrange(0, ninit): 
 
        kmeans = KMeans(init='k-means++', n_clusters=minK, n_init=1, max_iter=1)
        kmeans.fit(subs_matrix)
        centroids = kmeans.cluster_centers_
        normalize_centroids(centroids)
        for iter in xrange(1,maxiter):        
            kmeans = KMeans(init=centroids, n_clusters=minK, n_init=1, max_iter=1)                 
            kmeans.fit(subs_matrix)
            centroids = kmeans.cluster_centers_
            normalize_centroids(centroids)            
        inertia = kmeans.inertia_
        
        if best_centroids is None or inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
        
    for j in xrange(0,len(best_centroids)):        
        cluster_vec = [(i2w[i], weight) for (i, weight) in enumerate(best_centroids[j,:]) if weight != 0]
        cluster_vec = sorted(cluster_vec, key=itemgetter(1), reverse=True)[:cluster_prunning]
        norm = sum([weight**2 for word, weight in cluster_vec])**0.5
        cluster_vec = [(word, weight/norm) for word, weight in cluster_vec]
        norm = sum([weight**2 for word, weight in cluster_vec])**0.5
        cluster_file.write(target + "\t" + str(j) + "\t0\t" + target + "\tCLUSTER\t norm verified = " + '{0:1.8f}'.format(norm) + "\tpruning factor = " + str(cluster_prunning) +"\n")
        for (word, weight) in cluster_vec:
            cluster_file.write(' '.join([word, '{0:1.8f}'.format(weight)])+'\t')
        cluster_file.write('\n') 
    
    subvec_file.close()
    cluster_file.close()


if __name__ == '__main__':
    
    if len(sys.argv) < 9:
        sys.stderr.write("Usage: %s <vocab-file> <K> <min-avg-cluster-size> <cluster-prunning> <input-dir> <output-dir> <from> <to> [n_init] [max_iter]\n" % sys.argv[0])
        sys.exit(1)
        
    vocab_filename =  sys.argv[1]
    K = int(sys.argv[2])
    min_avg_cluster_size = int(sys.argv[3])
    cluster_prunning = int(sys.argv[4])
    input_dirname = sys.argv[5]
    output_dirname = sys.argv[6]
    from_file = int(sys.argv[7])
    to_file = int(sys.argv[8])
    
    if from_file == 0:
        from_file = None
    if to_file == 0:
        to_file = None
    w2i, i2w = load_vocabulary_w2i(vocab_filename)

    ninit=1
    maxiter=30
    if len(sys.argv) > 9:
        ninit = int(sys.argv[9])
    if len(sys.argv) > 10:
        maxiter = int(sys.argv[10])
    sys.stderr.write("K=%d, n_init=%d, max_iter=%d\n" % (K, ninit, maxiter))
    
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)
    
    filenames = sorted(os.listdir(input_dirname))[from_file:to_file]     
   
    for filename in filenames:
        input_filepath = '/'.join([input_dirname, filename])  
        output_filepath = '/'.join([output_dirname, filename])
        cluster_subvec_file(w2i, cluster_prunning, K, ninit, maxiter, min_avg_cluster_size, input_filepath, output_filepath)

    
    