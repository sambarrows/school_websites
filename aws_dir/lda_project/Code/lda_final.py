import gensim
import logging
from gensim import corpora

my_topics = 8
my_workers = 8
 
# To see output as code is running, uncomment line below
# logging.basicConfig(format='%(asctime)s: %(levelname)s : %(message)s', level=logging.INFO)

data_directory = '../Data/'
my_dict = corpora.Dictionary.load(data_directory + 'lda_dictionary.dict')
my_corpus = corpora.MmCorpus(data_directory + 'lda_corpus.mm')

my_chunksize = int(float(len(my_corpus))/my_workers)
my_lda = gensim.models.LdaMulticore(my_corpus, num_topics=my_topics, id2word=my_dict, passes=30, workers=my_workers, chunksize=my_chunksize)
my_lda.save('../Results/lda_8topics.lda')

topic_list  = my_lda.print_topics(num_topics=my_topics, num_words=8)
for t in topic_list:
    print t[1]

