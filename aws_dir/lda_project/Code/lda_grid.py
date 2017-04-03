import gensim
import random
import numpy as np
import pandas as pd
from gensim import corpora

## Set number topics
k = 10

## Set the parameter values that I will search over
param_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5]

# Read in dictionary and corpus
my_dictionary = corpora.Dictionary.load('../Data/lda_dictionary.dict')
my_corpus = corpora.MmCorpus('../Data/lda_corpus.mm')

# Split into 80% training and 20% test sets
cp = list(my_corpus)
random.shuffle(cp)
p = int(len(cp) * .8)
cp_train = cp[0:p]
cp_test = cp[p:]

# Compute perplexities
grid_vals = {}
for alpha_val in param_list:
    perp_vals = []
    for eta_val in param_list:
        lda = gensim.models.ldamodel.LdaModel(cp_train, num_topics=k, alpha=alpha_val, eta=eta_val, id2word=my_dictionary, passes=20)
        per_word_perplex = np.exp2(-lda.log_perplexity(cp_test))
        perp_vals.append(per_word_perplex)
    grid_vals[alpha_val] = perp_vals

# Convert dictionary to DataFrame, with alpha on x axis, and eta on y axis
final_grid = pd.DataFrame(grid_vals, index=param_list)

# Write grid of perplexities to csv
final_grid.to_csv('../Results/alpha_eta_grid.csv')
