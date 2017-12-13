# -*- coding: utf-8 -*-

from LDA import *
import pickle
lda = LDA()
lda.load()

f = open("./data/preprocessed_lyric.txt","r",encoding="utf-8")
doc = {}

for line in f:
    id_song = line.split(";")
    document_topics, word_topic, word_phi = lda.getTopic(lda.sen2bow(id_song[1]))
    doc[id_song[0]]= document_topics
f.close()

f2 = open("./data/ldamatrix.txt", "wb")

pickle.dump(doc, f2)

f2.close()

