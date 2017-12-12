# -*- coding: utf-8 -*-

from LDA import *
import json
lda = LDA()


lda.load()

f = open("./data/preprocessed_lyric.txt","r",encoding="utf-8")
doc = []

for line in f:
    id_song = line.split(";")
    doc.append(id_song[0])
    doc.append(id_song[1].split(","))
f.close()

f2 = open("./data/ldamatrix.txt", "wb")

data = {}
for song in doc:
    lyric = song[1]
    matrix = lda.sen2bow(lyric)
    document_topics, word_topic, word_phi = lda.getTopic(matrix)

    data[str(song[0])] = document_topics

json.dump(data, f2)

f2.close()

