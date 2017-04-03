import random
import gensim
import json
import numpy as np
from gensim import corpora

# Set the different numbers of topics that I will try
num_topic_list = range(2, 102, 10)

# Set number of cores to use when running gensim
my_workers=8

# Read in dictionary and corpus
my_dictionary = corpora.Dictionary.load('../Data/lda_dictionary.dict')
my_corpus = corpora.MmCorpus('../Data/lda_corpus.mm')

# Split into 80% training and 20% test sets
cp = list(my_corpus)
random.shuffle(cp)
p = int(len(cp) * .8)
cp_train = cp[0:p]
cp_test = cp[p:]

# Compute perplexities for different numbers of topics
perplexities = []
for n in num_topic_list:
    # fit model
    my_chunksize = int(float(len(my_corpus))/my_workers)
    lda = gensim.models.LdaMulticore(cp_train, num_topics=n, id2word=my_dictionary, passes=20, workers=my_workers, chunksize=my_chunksize)

    # calculate per-word perplexity
    per_word_perplex = np.exp2(-lda.log_perplexity(cp_test))

    # Alternative way of getting per-word perplexity
    # per_word_perplex = np.exp2(-lda.bound(cp_test) / sum(cnt for document in cp_test for _, cnt in document))

    perplexities.append(per_word_perplex)
    print n

topics_perps_lists = [num_topic_list, perplexities]

# Write lists of number of topics and corresponding perplexities to json
with open('../Results/topics_perps_lists.txt', 'w') as myfile:
    json.dump(topics_perps_lists, myfile)
