import os
import sys
from konlpy.tag import Twitter

target = 'ko'
model = Twitter()

input=open('output_modi.txt','r')
output=open('output_twitter.txt','w')

strlist = []
count = 0
for ss in input:
    count=count+1
    strlist.append(ss)

    if count!=10 :
        continue
       
    enWord = ss.split('\t')[0]
    koWord = model.pos(enWord,True,True)
    if koWord.count is 0:
        count=0
        strlist.clear()
        continue
    outt = ""
    for k in koWord:
        if k[1] != 'Josa':
            outt = outt+" "+k[0]

    for txt in strlist:
        output.writelines(txt.replace(enWord,outt.strip()))
        
    count=0
    strlist.clear()

input.close()
output.close()
