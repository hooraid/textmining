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

def RecommendSong(text):
    mynet = YSentiNet()
    text_score = mynet.GetTextScore(text,2)
    song_list = open("analysis.txt",'r')
    
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
    return ip/(asum*bsum)

def main():
    RecommendSong("뚜루뚜뚜뚜")
    return

if __name__ == "__main__":
    main()