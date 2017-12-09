# -*- coding: utf-8 -*-

import logging
import numpy
from matplotlib.pyplot import plot, show

from gensim import corpora
from gensim.models import LsiModel
from gensim import matutils

from sklearn.cluster import KMeans


dictionary = corpora.Dictionary.load("./tmp/dictionary.dict") # load dictionary
corpus = corpora.MmCorpus("./tmp/corpus.mm") # load corpus

#params for LSI
n_topics = 100

# LSI for dimension reduction
lsi_model = LsiModel(corpus, id2word=corpus.dictionary, num_topics=n_topics)
corpus_lsi = lsi_model[corpus]

corpus_lsi_dense = matutils.corpus2dense(corpus_lsi, n_topics)

# K means parameter setting
true_k = 0 # num of clusters

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
km.fit(corpus_lsi_dense)

##print str(km.labels_)
#labels = km.labels_      #<============WRONG

#print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_)
#print "Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_)
#print "V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_)
#print "Adjusted Rand-Index: %.3f" %\
#      metrics.adjusted_rand_score(labels, km.labels_)
#print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(
#    corpus_lsi_dense, labels, sample_size=1000)
#
#print
