import os
import sys
import numpy as np
import math
from konlpy.tag import Twitter

class YSentiNet(object):
    def __init__(self, filename = "output_twitter.txt"):
        self.net = self.MakeDict(filename=filename)
        self.model = Twitter()
        self.dictionary = filename
        self.size = self.net["size"]
        self.category = self.net["sentimental"]
        self.zerosenti = [0,0,0,0,0,0,0,0,0,0]

    def MakeDict(self,filename = "output_twitter.txt"):
        
        txt = open(filename, 'r')
        ret = {'filename':filename}
        ret['sentimental']='anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust'

        size = 0
        count = 0
        sent = []
        for tt in txt:
            count=count+1
            sent.append(int(tt.split('\t')[2]))
            if count != 10:
                continue

            if tt.split('\t')[0] in ret:
                count=0
                sent.clear()
                continue

            ret[(tt.split('\t')[0]).strip()] = sent.copy()
            sent.clear()
            count = 0
            size=size+1
        ret['size'] = size
        return ret

    def GetWordScore(self, inputword=""):
        inputword = self.TokenizeText(inputword)
        text=""
        for tt in inputword:
            text = text+" "+tt
        score = self.net.get(text.strip())
        if score is None:
            return self.zerosenti
        else:
            return score
    
    def GetTextScore(self, inputtext="", tokenlength=2):
        tokenized = self.TokenizeText(inputtext)
        score = self.zerosenti
        pprev_token = ""
        prev_token = ""
        for token in tokenized:
            single_score = self.GetWordScore(token)
            score = list(np.array(score)+np.array(single_score))
            if tokenlength==2 :
                twowords_score = self.GetWordScore(prev_token+" "+token)
                score = list(np.array(score)+np.array(twowords_score))
            if tokenlength==2 :
                threewords_score = self.GetWordScore(pprev_token+" "+prev_token+" "+token)
                score = list(np.array(score)+np.array(threewords_score))
            pprev_token = prev_token
            prev_token = token
        return score

    def TokenizeText(self, inputtext=""):
        text = self.model.pos(inputtext,True,True)
        output = []
        for token in text:
            if token[1] == 'Josa':
                continue
            output.append(token[0])
        return output
    
    def PrettyPrintScore(self, score=[0,0,0,0,0,0,0,0,0,0]):
        category = self.category.split(', ')
        for cat,sco in zip(category, score):
            print(cat+" : "+str(sco))
        return
    
    def Normalize(self, score=[0,0,0,0,0,0,0,0,0,0]):
        if score is self.zerosenti:
            return score
        sum = 0
        for sc in score:
            sum = sum+sc*sc
        sum = math.sqrt(sum)
        for idx in range(len(score)):
            score[idx] = round(score[idx]/sum,3)
        return score

#test
def test():
    test_lyrics = '''이번 학기도 결국은 언제나처럼 벼락치기로 공부하는데, 생각해보면 중간고사 이후로는 여러가지 프로젝트들이 상호배타적으로 나와서 이리 치이고 저리 치이고 하다가 결국 어영부영 벼락치기해서 기말고사를 보는 것이 패턴이 된 것 같다.
특히 이번 학기에는 유례 없는 4팀프로젝트+튜터링에 시달리고 결국 어떻게든 다 끝내놓고 나니까 결국 3~4일동안 3과목을 벼락치기해야 하는 상황에 직면했다. (참고로 현재는 저기서 2일 지남) 그럼에도 불구하고 (말로는 던진다 던진다 하고 딴짓은 많이 하면서) 기본적으로는 안던지려고 발악하는 걸 보면 성격은 고치기 힘들구나'''
    mynet = YSentiNet()
    print(mynet.dictionary)
    print(mynet.category)
    print(mynet.size)
    print("#### GetWordScore ####")
    text = ''.join(str(x)+" " for x in mynet.GetWordScore("사랑해"))
    print("score of 사랑해 : "+text)
    print("#### GetSongScore ####")
    score = mynet.GetTextScore(test_lyrics)
    text = ''.join(str(x)+" " for x in score)
    print("score of Song : "+text)
    mynet.PrettyPrintScore(mynet.Normalize(score))    

if __name__ == "__main__":
    test()