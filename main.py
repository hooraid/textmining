from clustering import LDA
from YSentiNet import YSentiNet
from YSentiNet import tfidf
import pickle
import operator
import numpy as np
import math

def testclustering():
    text = '''
머릿속이 복잡한 요즘
나를 가볍게 안아주며 잘할수 있다고 말해줄 수 있는 그런 사람이 있으면 좋겠다.
혼자 있기에는 너무 적적한데
그런 지금 나를 만나주는 사람들이 있다는 건 참 다행이다.
    '''
    yc = tfidf.YCluster(False)
    top5score, top5song = yc.GetTopFive(text)
    for idx in range(len(top5score)) :
        print("#\t"+str(idx+1))
        print("#    score   : "+str(top5score[idx]))
        print("#    artist  : " + top5song[idx].split(';')[1])
        print("#    title   : " + top5song[idx].split(';')[2])
        print()

    return

def MakeSentiScore():
    mynet = YSentiNet.YSentiNet()
    
    lyrics = open("YSentiNet/remove_lyrics.txt", 'r')
    output = open("YSentiNet/analysis.txt", 'w')

    count = 0
    for lyr in lyrics:
        tt = lyr.split(';')
        score = mynet.GetTextScore(tt[0])
        artist = tt[1]
        title = tt[2]
        output.write(str(score)+";"+artist+";"+title+"\n")
    
    print("fitnih")
    return

def RecommendSong(text):
    mynet = YSentiNet.YSentiNet()
    text_score = mynet.GetTextScore(text,2)
    song_list = open("YSentiNet/analysis.txt",'r')
    
    top5score = [-10,-10,-10,-10,-10]
    top5songinfo = ["","","","",""]

    for song in song_list:
        song_score = song.split(';')[0]
        song_score = song_score[1:-1]
        song_score = list(map(int,song_score.split(',')))
        cosSim = GetCosSimilarity(song_score, text_score)
        for idx in range(len(top5score)):
            if top5score[idx] < cosSim:
                top5score[idx] = cosSim
                top5songinfo[idx] = song
                break

    for songinfo in top5songinfo:
        print(songinfo)

    return

def GetCosSimilarity(a, b):
    if len(a) is not len(b):
        print("ERROR:different dimension vectors")
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
    if asum*bsum == 0:
        return 0
    return ip/(asum*bsum)

def testSenti():
    RecommendSong("뚜루뚜뚜뚜")
    return


def testLDA(MakeModel = False):

    # make model (모델생성용! 사용안할 시 주석처리)
    lda = LDA.LDA()
        
    if MakeModel:
        topic_n = 10 # 토픽 개수
        lda.makemodel(topic_n)
    lda.load()
    # get dense LDA matrix
    LDAmatrix = pickle.load(open("./clustering/data/ldamatrix.txt","rb"))

    input = '''
    길었던 새벽 별빛 아래 홀로
조금 취해버린 나의 맘과
너의 맘이 우리의 말이
어느새 갈 곳을 잃었는지
걷기엔 조금은 지쳤나 봐
아무도 없는 정류장에 앉아
너의 이름 불러본다
내게도 선명히 들려온다
꼭 잡은 손 마주 앉은 우리
함께 걷던 거리
손 내밀며 날 부른 소리
이른 새벽 지쳐있던 우릴 
밝혀준 별이
다가온다
이제는 부서진 맘 이기적인 난
네가 없는 첫차를 타고
참았던 눈물을 흘려본다
소리 내서 울어본다
되돌아가고 싶어 미쳐버린 난
어디론가 크게 외쳐봐도
소리 없이 네게서 떠나간다
처음 그곳 제자리로
꼭 잡은 손 마주 앉은 우리
함께 걷던 거리
손 내밀며 날 부른 소리
이른 새벽 지쳐있던 우릴 
밝혀준 별이
다가온다
이제는 부서진 맘 이기적인 난
네가 없는 첫차를 타고
참았던 눈물을 흘려본다
소리 내서 울어본다
되돌아가고 싶어 미쳐버린 난
어디론가 크게 외쳐봐도
소리 없이 네게서 떠나간다
처음 그곳 제자리로
되돌아가고 싶어 미쳐버린 난
어디론가 크게 외쳐봐도
소리 없이 네게서 떠나간다
처음 그곳 제자리로
    '''

    # # get similarity
    matrix = lda.sen2bow(input)

    document_topics, word_topic, word_phi = lda.getTopic(matrix)
    print(document_topics)
    print(LDAmatrix['30666642'])
    data={}
    for song in LDAmatrix.keys():

        sim_value = lda.getSim(document_topics, LDAmatrix[song])
        data[song] = sim_value

    sorted_data = sorted(data.items(), key = operator.itemgetter(1), reverse=True)

    songlist = []
    for k in range(0, 5):
        songlist.append(sorted_data[k])

    lyricf = open("./clustering/data/id-lyrics.txt","r",encoding="utf-8")

    id_lyric ={}
    for line in lyricf:
        doc = line.split(";")
        id_lyric[doc[0]] = doc[1]

    for id in songlist :
        print(id[0])
        print(id[1])
        print(id_lyric[id[0]])

class YNet(object):
    def __init__(self):
        self.mynet = YSentiNet.YSentiNet()
        self.yc = tfidf.YCluster(False)
        self.lda = LDA.LDA()
        self.lda.load()
        self.dict = self.MakeVectorMatrix(save=False,load=True)

    def GetSentiVector(self,input):
        return self.mynet.Normalize( self.mynet.GetTextScore(input,2) )

    def GetSentiTop(self,input):
        
        text_score = self.mynet.GetTextScore(input,2)
        song_list = open("YSentiNet/analysis.txt",'r')
        
        top5score = [-10,-10,-10,-10,-10]
        top5songinfo = ["","","","",""]

        for song in song_list:
            song_score = song.split(';')[0]
            song_score = song_score[1:-1]
            song_score = list(map(int,song_score.split(',')))
            cosSim = GetCosSimilarity(song_score, text_score)
            for idx in range(len(top5score)):
                if top5score[idx] < cosSim:
                    top5score[idx] = cosSim
                    top5songinfo[idx] = song
                    break

        for songinfo in top5songinfo:
            print(songinfo)

        return

    def GetClusterTop(self,input):
        top5score, top5song = self.yc.GetTopFive(input)
        for idx in range(len(top5score)) :
            print("#\t"+str(idx+1))
            print("#    score   : "+str(top5score[idx]))
            print("#    artist  : " + top5song[idx].split(';')[1])
            print("#    title   : " + top5song[idx].split(';')[2])
            print()
        return

    def GetLDATop(self,input,simple=True):
        LDAmatrix = pickle.load(open("./clustering/data/ldamatrix.txt","rb"))
        matrix = self.lda.sen2bow(input)
        document_topics, word_topic, word_phi = self.lda.getTopic(matrix)

        data={}
        for song in LDAmatrix.keys():

            sim_value = self.lda.getSim(document_topics, LDAmatrix[song])
            data[song] = sim_value

        sorted_data = sorted(data.items(), key = operator.itemgetter(1), reverse=True)

        songlist = []
        for k in range(0, 5):
            songlist.append(sorted_data[k])

        lyricf = open("./clustering/data/id-lyrics.txt","r",encoding="utf-8")

        id_lyric ={}
        for line in lyricf:
            doc = line.split(";")
            id_lyric[doc[0]] = doc[1]

        print("######################################################")
        print("**LDA Top 5 Score**")
        print()
        for idx in range(5):
            print("#"+str(idx+1))
            print("score    : "+str(songlist[idx][1]))
            print("id       : "+str(songlist[idx][0]))
            if not simple:
                print("lyrics   : "+id_lyric[songlist[idx][0]])
            print()    
        print("######################################################")
        return

    def GetClusterVector(self,input):
        return self.yc.GetClusterVector(input)

    def GetLDAVector(self,input):
        matrix = self.lda.sen2bow(input)
        document_topics, word_topic, word_phi = self.lda.getTopic(matrix)
        return self.lda.GetVec(document_topics)

    def GetLyrVec(self, input):
        t = []
        t[0:9] = self.GetSentiVector(input)
        t[10:19] = self.GetLDAVector(input)
        t[20:29] = self.GetClusterVector(input)
        return t

    def MakeVectorMatrix(self, save=False, load=False):
        if load:
            return pickle.load(open("VecMat.pickle",'rb'))
        
        f = open("clustering/data/remove_lyrics.txt")
        
        Matrix = {}

        for line in f:
            t = []
            songid = line.split(';')[-1]
            lyrics = line.split(';')[0]
            t[0:9] = self.GetSentiVector(lyrics)
            t[10:19] = self.GetLDAVector(lyrics)
            t[20:29] = self.GetClusterVector(lyrics)
            Matrix[songid] = t

        if save:
            pickle.dump(Matrix,open("VecMat.pickle", 'wb'))

        return Matrix

    def GetTopFive(self, input, simple=True):
        lyrVec = self.GetLyrVec(input)
        top5score = [0,0,0,0,0]
        top5song = ["","","","",""]
        for k,v in self.dict.items():
            sim = GetCosSimilarity(lyrVec, v)
            for idx in range(5):
                if top5score[idx] < sim :
                    top5score[idx] = sim
                    top5song[idx] = k
                    break
        zz = zip(top5score, top5song)
        zz = sorted(zz, key=lambda x:x[0], reverse=True)
        
        top5score, top5song = zip(*zz)


        lyricf = open("./clustering/data/id-lyrics.txt","r",encoding="utf-8")
        id_lyric ={}
        for line in lyricf:
            doc = line.split(";")
            id_lyric[doc[0]] = doc[1]
        
        print("######################################################")
        print("**Total Top 5 Score**")
        print()
        for idx in range(5):
            print("#"+str(idx+1))
            print("score    : "+str(top5score[idx]))
            print("id       : "+str(top5song[idx][:-1]))
            if not simple:
                print("lyrics   : "+id_lyric[top5song[idx][:-1]])
            print()
        print("######################################################")

if __name__ == '__main__':
    inputtext = '''길었던 새벽 별빛 아래 홀로 조금 취해버린 나의 맘과 너의 맘이 우리의 말이 어느새 갈 곳을 잃었는지 걷기엔 조금은 지쳤나 봐 아무도 없는 정류장에 앉아 너의 이름 불러본다 내게도 선명히 들려온다 꼭 잡은 손 마주 앉은 우리 함께 걷던 거리 손 내밀며 날 부른 소리 이른 새벽 지쳐있던 우릴 밝혀준 별이 다가온다 이제는 부서진 맘 이기적인 난 네가 없는 첫차를 타고 참았던 눈물을 흘려본다 소리 내서 울어본다 되돌아가고 싶어 미쳐버린 난 어디론가 크게 외쳐봐도 소리 없이 네게서 떠나간다 처음 그곳 제자리로 꼭 잡은 손 마주 앉은 우리 함께 걷던 거리 손 내밀며 날 부른 소리 이른 새벽 지쳐있던 우릴 밝혀준 별이 다가온다 이제는 부서진 맘 이기적인 난 네가 없는 첫차를 타고 참았던 눈물을 흘려본다 소리 내서 울어본다 되돌아가고 싶어 미쳐버린 난 어디론가 크게 외쳐봐도 소리 없이 네게서 떠나간다 처음 그곳 제자리로 되돌아가고 싶어 미쳐버린 난 어디론가 크게 외쳐봐도 소리 없이 네게서 떠나간다 처음 그곳 제자리로'''
    ynet = YNet()
    #vecMat = ynet.MakeVectorMatrix(save=False, load=True)
    print(ynet.GetSentiVector(inputtext))
    print(ynet.GetLDAVector(inputtext))
    print(ynet.GetClusterVector(inputtext))
    ynet.GetClusterTop(inputtext)
    ynet.GetLDATop(inputtext)
    ynet.GetSentiTop(inputtext)
    ynet.GetTopFive(inputtext)
    return
