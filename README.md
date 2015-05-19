WORD2PARVEC TOOLKIT		Oren Melamud, 2015
-------------------------------------------

NOTE: The word2parvec toolkit is provided 'AS IS' with no warranty whatsoever.

word2parvec is a toolkit that learns paraphrase vector (parvec) representations for word meanings in context.
The model is described in the following paper (please cite if using this toolkit):

Oren Melamud, Ido Dagan, Jacob Goldberger. Modeling Word Meaning in Context with Substitute Vectors. NAACL, 2015.

This readme file explains how to use the toolkit.
The procedure includes the following steps:

SETUP

	1. Preprocessing the learning corpus
	2. Learning an n-gram language model from the corpus
	3. Extracting sentential contexts from the corpus for all target words
		a. Choosing target words
		b. Sampling target words contexts
		c. Generating substitute vector (subvec) representations for contexts
INFERENCE

	4. Generating parvecs for target words in sentential context

The toolkit also includes:

	5. A simple Wordnet-based pseudoword generator
	6. An application that evaluates subvec/bow/cbow context similarity measures using pseudowords
	

1. Preprocessing the learning corpus
-------------------------------------
This is a common procedure in many NLP tasks. Use your favorite tools to perform the following steps:
	1.1 Sentence split - one sentence per line
	1.2 Tokenize - space-delimited tokens in each line
	Optional:
	1.3 Convert all words to lowercase
	1.4 Convert rare words to special token (e.g. <RW>)
	1.5 Convert numbers to special token (e.g. <NUM>)
	1.6 Shuffle the lines of the corpus to avoid unintentional bias to corpus structure

We denote the preprocessed learning corpus file as CORPUS.
Finally, use the following script to generate a vocabulary file, denoted VOCAB, for the corpus:

	cat CORPUS | python count_vocab.py 0 > VOCAB


2. Learning an n-gram language model from the corpus
----------------------------------------------------
There are several n-gram language model toolkits.
You can use any toolkit that can export the leanred language model into a standard ARPA format.
We denote the language model ARPA file as LM.arpa

KenLM is one good option:
	You can download this toolkit from https://kheafield.com/code/kenlm/ and follow the instructions.
	An example command line for learning a 5-gram Kneser Ney language model is:
	bin/lmplz -o 5 -S 48G -T ~/tmp --text CORPUS --prune 0 2 2 2 2 > LM.arpa


3. Extracting sentential contexts from corpus for all target words
---------------------------------------------------------------

3.a. Choosing target words
-------------------------
Create a file with one word per line comprising all of the target words that you will need in your application.
We denote the target file as TARGETS

Note that you will need to allocate sufficient disk space for storing the contexts that will be collected from the corpus for each of the targets (~20M per each target word type).


3.b. Sampling target words contexts
----------------------------------
Sample sentential contexts for all of your target words using the script below.
	
	python extract_contexts.py CORPUS TARGETS <contexts-num> TARGETS_CONTEXTS TARGETS_FREQS

	TARGETS_CONTEXTS denotes a file containing the corpus contexts sampled for the targets (this can be a very big file)
	TARGETS_FREQS denotes a file containing the number of sampled contexts per each target word type 
	<contexts-num> is the maximum number of contexts sampled per each target (e.g. 20000)


3.c. Generating substitute vector (subvec) representations for contexts
----------------------------------------------------------------------

(i) Generating fastsub subvecs

	To compute subvecs for the target words contexts use the FASTSUBS toolkit.
	Download FASTSUBS from: https://github.com/ai-ku/fastsubs and use as follows:

	cat TARGETS_CONTEXTS | ./fastsubs-omp -n <pruning-factor> -m <thread-num> -t -z LM.arpa  > TARGETS_SUBVECS

	<pruning-factor> is the maximum number of entries in each subvec (suggested value 100)
	<thread-num> is the maximum number of threads that fastsubs-omp will use on your machine
	TARGETS_SUBVECS is the targets context file with subvec representations (this would be an even bigger file...)

(ii) Optional (recommended) context cleanup:

	The following script extracts only contexts where the original target that was observed with this context appears in the subvec:

	python extract_reliable_subvecs.py TARGETS_SUBVECS TARGETS_SUBVECS.RELIABLE TARGETS_FREQS.RELIABLE

(iii) Converting subvecs weights from conditional probability to SPPMI:

	cat TARGETS_SUBVECS.RELIABLE | python subvecs2pmi.py VOCAB  <shift> > TARGETS_SUBVECS.RELIABLE.PMI

	<shift> is the sppmi shift parameter (recommended value: 2.0)

(iv) Converting the large contexts file to a directory of files:

	This script converts the big contexts subvec file to a more application-friendly file-per-target directory.
	This will create the directory TARGETS_SUBVECS.RELIABLE.PMI.DIR with a file named w for every target word type w in TARGETS_SUBVECS.

	python subvec_dir.py  TARGETS_SUBVECS.RELIABLE.PMI

(v) Clustering subvecs - Optional

	The following script clusters contexts together in order to reduce the size of the target subvecs directory.

	cluster_subvecs_concurrently.sh <source-home> <process-num> VOCAB <cluster-num> 1 <cluster-prunning> TARGETS_SUBVECS.RELIABLE.PMI.DIR TARGETS_SUBVECS.RELIABLE.PMI.CLUSTER.DIR [<n_init>] [<max_iter>]

	<source-home> is the directory under which the python source code parvecs is installed
	<process-num> number of processes spawned concurrently
	<cluster-num> is the number of context cluster per each word type
	<cluster-prunning> is the max number of entries in cluster vectors
	<n_init> is the number of different random starting points for the clustering process (default 1)
	<max_iter> is the max number of iterations performed in the clustering process (default 30)

	Note: the output cluster subvecs are L2-normalized


4. Generating parvecs for target words in sentential context
---------------------------------------------------------------
To compute parvecs, your words-in-contexts file, denoted TEST, should be formatted in the same way as in the file TARGETS_CONTEXTS from section 3.b.
The follow the instructions in 3.c.(i) and 3.c.(iii) to generate substitute vectors for your test file TEST.SUBVECS.PMI.
In this file there should be for every target word instance two lines:
target_name <tab> target_id <tab> target_index <tab> text_line
sub1 <space> weight1 <tab> sub2 <space> weight2 <tab> ...

The substitutes in the second line are for the target at text_line[target_index] (i.e. the word in the target_index position in text_line).

Note:
- To speed up parvec generation considerably, sort the contexts in TEST according to their target_name (i.e. contexts of the same target word should be grouped together).
- It is generally recommended to use same subvec weighting schemes (e.g. PMI with a shift of 2.0) for both TARGETS_SUBVECS.RELIABLE.PMI and TEST.SUBVECS.PMI. 

To generate parvecs for words in context run:

	python word2parvec.py -contexts_dir TARGETS_SUBVECS.RELIABLE.PMI.DIR -vocabfile VOCAB -testfile TEST.SUBVECS.PMI -resultsfile TEST.PARVECS
	or
	python word2parvec.py -contexts_dir TARGETS_SUBVECS.RELIABLE.PMI.CLUSTER.DIR --excluderef -vocabfile VOCAB -testfile TEST.SUBVECS.PMI -resultsfile TEST.PARVECS

	TEST.PARVECS is the output file that will be created with the following 3 lines for every target word instance:
	INSTANCE <tab> target_name <tab> target_id <tab> target_index <tab> text_line
	SUBVEC <tab> sub1 <space> weight1 <tab> sub2 <space> weight2 <tab> ...
	PARVEC <tab> par1 <space> weight1 <tab> par2 <space> weight2 <tab> ...

You can use the following runtime arguments:

	--excluderef excludes the given context from the target contexts average. This is recommended when using clustered subvecs (3.c.(v)).

	--lemmatize can be used to convert the parvec to lemmatized form (useful, for istance, when evaluating against a gold standard of lemmas). 
	When using this option the target_name in TEST should be in the form of <string>.POS where POS is a wordnet part-of-speech identifier (ADJ, ADV, NOUN, VERB = 'a', 'r', 'n', 'v').
	A 4th line will be included in the output:
	PARLEMVEC <tab> parlem1 <space> weight1 <tab> parlem2 <space> weight2 <tab> ...

	-top <int> and -toppercent <percent> can be used to inject a stronger bias in the parvec towards the given context by averaging only on the top target contexts that are most similar to the given context.

	-weightsfactor <float> sets a float value f. The context similarity function is implemented as sim(c1,c2) = cos(c1,c2)^f, where the default value of f is 1.0

	-parvec_maxlen <int> can be used to limit the number of entries in the generated parvecs

	--debug turns debug logs on
	--debugtop <int> limits the number of entries printed per vector

To generate parvecs for words out-of-context use: -weightsfactor 0.0 --excluderef


5. Pseudoword generator
------------------------
To randomly generate pseudowords run:

	wn_pseudoword_generator VOCAB <words-num> <words2senses-file> <senses-file> [<min-freq>]

	<words-num> is the number of pseudowords to be generated

	<words2senses-file> is an output file, denoted WORDS2SENSES, containing a single line for every pseudoword in the following format:
	pseudoword_name <tab> sense_word1 <space> sense_word2 <space> ...

	<senses-file>, denoted SENSES, is an output file with all of the senses from all pseudowords (one sense per line)
	<min-freq> is the minimum corpus frequency for a sense to be acceptable (default value is 1000).


6. Context similarity measures evaluation
-------------------------------------------
Performs steps 3.b and 3.c.(i) and 3.c.(iv) using SENSES as TARGETS (<contexts-num> can be set to ~1000) to collect contexts for the pseudo-sense words into PSEUDO_TARGETS_SUBVECS.DIR.

Run the following script to evaluate subvec (SUB) similarity with conditional probabilities weights:

	python context_similarity_measures_eval.py -samplenum 100 -toppercent 0.01 -pseudosfile WORDS2SENSES -contexts_dir PSEUDO_TARGETS_SUBVECS.DIR -vocabfile VOCAB -resultsfile <results>

To evaluate SUB sppmi weights add the following params:
	--pmi 
	-pmioffset <shift>

To evaluate with bag-of-words (BOW) context similarities add the following params:
	-bowinter 1.0
	-bow <window-size>
	(window size zero means the entire sentence)

To evaluate with continuous bow (CBOW):
	
	Use word2vec (https://code.google.com/p/word2vec/) to learn word embeddings.
	An example command line:
	./word2vec -train CORPUS -output EMBEDDINGS -cbow 0 -size 600 -window 4 -negative 15 -threads 12 -binary 0 -min-count 100
	
	The following script will convert the format of the embeddings
	python embedding_text2numpy.py EMBEDDINGS
	
	Add the following param to context_similarity_measures_app.py:
	-cbow EMBEDDINGS
	
To weigh the words in BOW/CBOW with tfidf weighting:
	--tfidf
	
To evaluate the combined SUB*CBOW measure (interpolation between SUB and CBOW measures), include both SUB and BOW/CBOW config params and use -bowinter 0.5	
	--debug turns debug logs on

	
	
