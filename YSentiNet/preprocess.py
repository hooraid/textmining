import os
import re
from YSentiNet import *

def preprocess_dictionary():
    input=open('output5.txt','r')
    output=open('output6.txt','w')

    strlist = []
    count = 0
    for ss in input:
        count=count+1
        strlist.append(ss)

        if count!=10 :
            continue
        
        enWord = ss.split('\t')[0]
        #koWord = re.sub(r'\[[^]]*\]', '', re.sub(r'\([^)]*\)', '',enWord))
        #koWord = re.sub(r'\{[^}]*\}', '', koWord);    
        koWord = enWord.split('(')[0]
        for txt in strlist:
            output.writelines(txt.replace(enWord,koWord))
            
        count=0
        strlist.clear()

    input.close()
    output.close()
    return

def preprocess_lyrics():
    model = YSentiNet()

    input=open('remove_lyrics.txt', 'r', encoding='UTF8')
    output=open('tokenized_lyrics.txt','w', encoding='UTF8')
    
    for lyr in input:
        tt = lyr.split(';')
        tt[0] = model.TokenizeText(tt[0])
        result = ""
        for tok in tt:
            result= result+ ';'+str(tok)
        output.writelines(result[1:])


if __name__ == "__main__":
    preprocess_lyrics()