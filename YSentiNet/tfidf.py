from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from konlpy.tag import Twitter 
import numpy as np
import pandas as pd
import os
import sys
import pickle
import math

'''
    cluster similarity : cosine between two 10-dim vectors
    10-dim vector : each element is distance to cluster center

'''


class YCluster(object):
    def __init__(self, create=False, n_cluster=10):
        self.corpus, self.song_info = self.MakeCorpus()        
        if create :
            self.tfidf_vectorizer, self.tfidf_matrix, self.km = self.Create_tfidf_cluster(self.corpus, n_cluster)
        else :
            self.tfidf_vectorizer, self.tfidf_matrix, self.km = self.Load_tfidf_cluster()
        if create : 
            self.dictionary = self.MakeClusterDictionary()
        else:
            self.dictionary = self.LoadClusterDictionary()
        self.prepro = Twitter()

    def MakeCorpus(self):
        f = open("YSentiNet/tokenized_lyrics.txt", 'r')    
        corpus = []
        song_info = []
        for line in f:
            words = line.split(';')
            toks = words[0].split(',')
            lyr=""
            for tok in toks:
                lyr = lyr + " " + tok[2:-1]
            lyr.strip()
            corpus.append(lyr[1:-1])
            song_info.append(line[len(words[0]):] )

        return corpus, song_info

    def Create_tfidf_cluster(self, corpus, clusters = 10):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,min_df=0.01,use_idf=True, ngram_range=(1, 3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        pickle.dump(tfidf_vectorizer, open("YSentiNet/tfidf_vectorizer.pickle", "wb"))
        pickle.dump(tfidf_matrix, open("YSentiNet/tfidf_matrix.pickle", "wb"))    

        num_clusters = clusters
        km = KMeans(n_clusters=num_clusters)
        km.fit(tfidf_matrix)
        pickle.dump(km, open("YSentiNet/km.pickle", "wb"))
        return tfidf_vectorizer, tfidf_matrix, km

    def Load_tfidf_cluster(self):
        tfidf_vectorizer = pickle.load(open("YSentiNet/tfidf_vectorizer.pickle", "rb"))
        tfidf_matrix = pickle.load(open("YSentiNet/tfidf_matrix.pickle", "rb"))
        km = pickle.load(open("YSentiNet/km.pickle","rb"))
        return tfidf_vectorizer, tfidf_matrix, km

    def GetClusterVector(self, text):
        centers = self.km.cluster_centers_
        tt = []
        tt.append(text)
        input = self.tfidf_vectorizer.transform(tt)
        input = input.toarray()[0]
        if centers.shape[1] != len(input):
            print("ERROR:different dimension vectors")
            return
        outVec = []
        for center in centers:
            outVec.append( self.GetDistance(center,input) )
        sum = 0
        for v in outVec:
            sum += v*v
        sum = math.sqrt(sum)     
        for idx in range(len(outVec)):
            outVec[idx] /= sum
        return outVec

    def MakeClusterDictionary(self):
        ret_dict = []
        centers = self.km.cluster_centers_
        for entry in self.tfidf_matrix.toarray():
            outVec = []
            for center in centers:
                outVec.append( self.GetDistance(center,entry) )
            sum = 0
            for v in outVec:
                sum += v*v
            sum = math.sqrt(sum)     
            for idx in range(len(outVec)):
                outVec[idx] /= sum
            ret_dict.append(outVec)
        pickle.dump(ret_dict, open("YSentiNet/clu_dict.pickle","wb"))
        return ret_dict

    def LoadClusterDictionary(self):
        ret = pickle.load(open("YSentiNet/clu_dict.pickle","rb"))
        return ret;

    def GetDistance(self,a,b):
        if len(a) != len(b):
            print("ERROR:GD, different dimension vectors")
            return -1
        sum = 0
        for idx in range(len(a)):
            sum += (a[idx]-b[idx])*(a[idx]-b[idx])
        return math.sqrt(sum)

    def GetCosSimilarity(self, a, b):
        if len(a) is not len(b):
            print("ERROR:GC, different dimension vectors")
            return -1
        ip = 0
        asum=0
        bsum=0
        for idx in range(len(a)):
            ip = ip+a[idx]*b[idx]
            asum = asum+a[idx]*a[idx]
            bsum = bsum+b[idx]*b[idx]
        asum = math.sqrt(asum)
        bsum = math.sqrt(bsum)
        return ip/(asum*bsum)
    
    def GetTopFive(self, text) :
        
        proc_text = self.TokenizeText(text)

        in_vec = self.GetClusterVector(proc_text)
        top5score = [0,0,0,0,0]
        top5song = ["","","","",""]
        for idx_vec in range(len(self.dictionary)):
            score = self.GetCosSimilarity(self.dictionary[idx_vec],in_vec)
            for idx in range(5):
                if top5score[idx] < score :
                    top5score[idx] = score;
                    top5song[idx] = self.song_info[idx_vec]
                    break
        return top5score, top5song
                    
    def TokenizeText(self, inputtext=""):
        text = self.prepro.pos(inputtext,True,True)
        output = ""
        for token in text:
            if token[1] == 'Josa':
                continue
            output += " "+(token[0])
        return output    
    
def test():
    text = '''
머릿속이 복잡한 요즘
나를 가볍게 안아주며 잘할수 있다고 말해줄 수 있는 그런 사람이 있으면 좋겠다.
혼자 있기에는 너무 적적한데
그런 지금 나를 만나주는 사람들이 있다는 건 참 다행이다.
    '''
    yc = YCluster(False)
    top5score, top5song = yc.GetTopFive(text)
    for idx in range(len(top5score)) :
        print("#\t"+str(idx+1))
        print("#    score   : "+str(top5score[idx]))
        print("#    artist  : " + top5song[idx].split(';')[1])
        print("#    title   : " + top5song[idx].split(';')[2])
        print()

    return

if __name__ == "__main__":
    test()