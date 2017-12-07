import os

input=open('lyrics.txt','r')
output=open('remove_lyrics.txt','w')

#가사 가수 제목
count = 0
for tt in input:
    if len(tt.split(';')) <2:
        continue
    title = tt.split(';')[2]
    if "inst" in title:
        continue
    if "Inst" in title:
        continue
    if "Ver." in title:
        continue
    if "ver." in title:
        continue
    if ".ver" in title:
        continue
    if ".Ver" in title:
        continue
    if "Remaster" in title:
        continue
    if "remaster" in title:
        continue
    if "Intro." in title:
        continue
    if "intro." in title:
        continue
    if "Outro." in title:
        continue
    if "outro." in title:
        continue
    output.writelines(tt)
    count=count+1

print(count)
