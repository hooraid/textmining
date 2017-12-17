import os
import sys
import numpy as np
import math
from konlpy.tag import Twitter

class YSentiNet(object):
    def __init__(self, filename = "YSentiNet/output_twitter.txt"):
        self.net = self.MakeDict(filename=filename)
        self.model = Twitter()
        self.dictionary = filename
        self.size = self.net["size"]
        self.category = self.net["sentimental"]
        self.zerosenti = [0,0,0,0,0,0,0,0,0,0]

    def MakeDict(self,filename = "YSentiNet/output_twitter.txt"):
        
        txt = open(filename, 'r', encoding='UTF8')
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
        if sum==0:
            return self.zerosenti
        for idx in range(len(score)):
            score[idx] = round(score[idx]/sum,3)
        return score

#test
def test():
    test_lyrics = '''
머릿속이 복잡한 요즘
나를 가볍게 안아주며 잘할수 있다고 말해줄 수 있는 그런 사람이 있으면 좋겠다.
혼자 있기에는 너무 적적한데
그런 지금 나를 만나주는 사람들이 있다는 건 참 다행이다.
'''
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