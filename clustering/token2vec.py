# -*- coding: utf-8 -*-

import os
import sys
from preprocess import  *

from gensim import corpora, models

class token2vec():
    def __init__(self, filename = ""):
        self.filename = filename

    def loadTokenfile(self):
        document_file = open(self.filename,"r", encoding="UTF-8") # need try~catch
        docfile2 = []
        for line in document_file:
            line.replace(";", ",")
            docfile2.append(line)
        return docfile2

    def sen2doc(self, sen):
        pre = preprocess()
        tokens = pre.pos4sen(sentence=sen)
        wrap = []
        wrap.append(tokens)
        return wrap

    def getDocTermMatrix(self, docfile2, condition = 1):
        self.condition = condition # 1 = id+tokens, 2 = only tokens
        if condition == 1 :
            docs = []
            for line in docfile2:
                tokens = line.split(",")
                docs.append(tokens)
            return docs

        elif condition == 2 :
            docsNoSongid = []
            for line in docfile2:
                tokens = line.split(",")
                docsNoSongid.append(tokens[1:])
            return docsNoSongid

    def saveDic(self, directory,docsNoSongid):
        dictionary = corpora.Dictionary(documents=docsNoSongid)
        dictionary.save(directory) # save the dictionary
        return dictionary

    def calculateTF(self, docs):
        '''
        :param docs:  2 dimension array with docs[1] == songid and docs[2] == tokens(array)
        :return:TF matrix (
        '''

        songTF = []
        for tokens in docs:
            temp =[]
            TFlist = {}
            temp.append(tokens[0])
            for token in tokens[1:]:
                if TFlist.__contains__(token):
                    TFlist[token] += 1
                else:
                    TFlist[token] = 1
            temp.append(TFlist)
            songTF.append(temp)

        return songTF

    def createBoW(self, songTF, dictionary):
        '''
        :param songTF: the return value of above method
        :param dictionary: also
        :return: corpus(that is 2 dimension array)
        '''

        corpus = []
        for song in songTF:
            tmp =[]
            temp = list(song[1].keys())
            for token in temp:
                tmp_tuple = (dictionary.token2id[token], song[1][token])
                tmp.append(tmp_tuple)
            corpus.append(tmp)
        return corpus

    def saveBow(self, corpus, directory):
        corpora.MmCorpus.serialize(directory, corpus) # save bag of words

    def calculateTfIdf(self, corpus):
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    def svaeTfIdf(self, corpus_tfidf, directory):
        corpora.MmCorpus.serialize(directory, corpus_tfidf) # save tfidf list
