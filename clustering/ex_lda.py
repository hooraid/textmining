from LDA import *
import pickle
import operator
import numpy as np
from konlpy.tag import Twitter

def main(MakeModel = False):

    # make model (모델생성용! 사용안할 시 주석처리)
    lda = LDA()
        
    if MakeModel:
        topic_n = 10 # 토픽 개수
        lda.makemodel(topic_n)
    lda.load()
    # get dense LDA matrix
    LDAmatrix = pickle.load(open("./data/ldamatrix.txt","rb"))

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

    lyricf = open("./data/id-lyrics.txt","r",encoding="utf-8")

    id_lyric ={}
    for line in lyricf:
        doc = line.split(";")
        id_lyric[doc[0]] = doc[1]

    for id in songlist :
        print(id[0])
        print(id[1])
        print(id_lyric[id[0]])


if __name__ == '__main__':
    main()