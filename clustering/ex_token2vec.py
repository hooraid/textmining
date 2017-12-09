# -*- coding: utf-8 -*-

import os
import sys

from token2vec import *

# load preprocessed lyric file
filename = "./test.txt"
test = token2tfidf(filename)
test.loadTokenfile()
documents = test.getDocTermMatrix(1)
documents_NoID = test.getDocTermMatrix(2)

# make dictionary(mapping token - id(term id)
dic_filename = "C:/Users/jiyun/Desktop/textmining/tmp/dictionary.dict"
dictionary = test.saveDic(dic_filename, documents_NoID)

#calculate term frequency
tfmatrix = test.calculateTF(documents)

# make corpus(same as bag of words)
corpus = test.createBoW(tfmatrix, dictionary)
corpus_filename = "C:/Users/jiyun/Desktop/textmining/tmp/corpus.mm"
test.saveBow(corpus_filename)

# make TF-IDF matrix
tfidf_matrix = test.calculateTfIdf(corpus)
tfidf_filename ="C:/Users/jiyun/Desktop/textmining/tmp/tfidf.mm"
test.svaeTfIdf(tfidf_filename)
