f = open("./data/lyrics.txt","r", encoding="utf-8")f2 = open("./data/id-lyrics.txt","w",encoding="utf-8")for line in f:    tmp = line.split(";")    lyric = tmp[0]    if len(tmp)<6 :        continue    id = tmp[6].replace("\n","")    str = id+";"+lyric+"\n"    f2.write(str)f2.close()f.close()