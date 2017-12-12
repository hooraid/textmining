#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

#load from kobill
from konlpy.corpus import kobill
docs_ko = [kobill.open(i).read() for i in kobill.fileids()]
#Tokenize
from konlpy.tag import Twitter; t = Twitter()
pos = lambda d: ['/'.join(p) for p in t.pos(d)]
texts_ko = [pos(doc) for doc in docs_ko]

from gensim import corpora
dictionary_ko = corpora.Dictionary(texts_ko)
dictionary_ko.save('ko.dict')  # save dictionary to file for future use

from gensim import models
tf_ko = [dictionary_ko.doc2bow(text) for text in texts_ko]
tfidf_model_ko = models.TfidfModel(tf_ko)
tfidf_ko = tfidf_model_ko[tf_ko]
corpora.MmCorpus.serialize('ko.mm', tfidf_ko) # save corpus to file for future use


lda_ko = models.ldamodel.LdaModel(tfidf_ko, id2word=dictionary_ko, num_topics=30)
#print(lda_ko.print_topics(num_topics=ntopics, num_words=nwords))

bow = tfidf_model_ko[dictionary_ko.doc2bow(texts_ko[0])]

sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)

bow = tfidf_model_ko[dictionary_ko.doc2bow(texts_ko[1])]

sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)
