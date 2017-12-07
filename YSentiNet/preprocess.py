import os
import re

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