from YSentiNet import *
import os,sys
import math


def MakeSentiScore():
    mynet = YSentiNet()
    
    lyrics = open("remove_lyrics.txt", 'r')
    output = open("analysis.txt", 'w')

    count = 0
    for lyr in lyrics:
        tt = lyr.split(';')
        score = mynet.GetTextScore(tt[0])
        artist = tt[1]
        title = tt[2]
        output.write(str(score)+";"+artist+";"+title+"\n")
    
    print("fitnih")
    return

def RecommendSong():
    mynet = YSentiNet()
    text = '''
관(棺)이 내렸다.
깊은 가슴 안에 밧줄로 달아내리듯 
주여 
용납하옵소서 
머리맡에 성경을 얹어주고 
나는 옷자락에 흙을 받아 
좌르르 하직했다. 

그 후로 
그를 꿈에서 만났다. 
턱이 긴 얼굴이 나를 돌아보고 
형(兄) 님!
불렸다. 
오오냐 나는 전신으로 대답했다. 
그래도 그는 못 들었으리라
이제 
네 음성을 
나만 듣는 여기는 눈과 비가 오는 세상. 

너는 어디로 갔느냐
그 어질고 안쓰럽고 다정한 눈짓을 하고 
형님!
부르는 목소리는 들리는데 
내 목소리는 미치지 못하는 
다만 여기는 
열매가 떨어지면 
툭 하고 소리가 들리는 세상.



'''
    text_score = mynet.GetTextScore(text,2)
    song_list = open("analysis.txt",'r')
    
    top5score = [-10,-10,-10,-10,-10]
    top5songinfo = ["","","","",""]

    for song in song_list:
        song_score = song.split(';')[0]
        song_score = song_score[1:-1]
        song_score = list(map(int,song_score.split(',')))
        cosSim = GetCosSimilarity(song_score, text_score)

        print(cosSim)

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
    return ip/(asum*bsum)

def main():
    RecommendSong()
    return

if __name__ == "__main__":
    main()