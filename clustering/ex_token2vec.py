# -*- coding: utf-8 -*-

import os
import sys

from token2vec import *

# load preprocessed lyric file
filename = "./test.txt"
test = token2vec(filename)
docfile2 = test.loadTokenfile()
documents = test.getDocTermMatrix(docfile2, 1)
documents_NoID = test.getDocTermMatrix(docfile2, 2)

# make dictionary(mapping token - id(term id)
dic_filename = "./tmp/dictionary.dict"
dictionary = test.saveDic(dic_filename, documents_NoID)

#calculate term frequency
tfmatrix = test.calculateTF(documents)

# make corpus(same as bag of words)
corpus = test.createBoW(tfmatrix, dictionary)
corpus_filename = "./tmp/corpus.mm"
test.saveBow(corpus, corpus_filename)

# make TF-IDF matrix
tfidf_matrix = test.calculateTfIdf(corpus)
tfidf_filename ="./tmp/tfidf.mm"
test.svaeTfIdf(tfidf_matrix, tfidf_filename)
