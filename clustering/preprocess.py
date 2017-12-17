# -*- coding: utf-8 -*-

from konlpy.tag import Twitter
import re

class preprocess() :
    original = []
    tagger = None
    stopwords = []

    def __init__(self, filename=""):
        self.filename = filename
        self.StopwordDic("./clustering/data/stopwords_kr.txt")

    def loadlyric(self) :
        f = open(self.filename, "r", encoding="utf-8")
        for line in f:
            lines = line.split(";")
            tmp = list()
            if(len(lines) > 1):
                tmp.append(lines[6])
                tmp.append(lines[0])
                self.original.append(tmp)
            else:
                continue
        f.close()

    def postagging(self, mode = "twitter") :
        if mode == "twitter":
            self.tagger = Twitter()
        output = []
        for song in self.original:
            lyric = song[1][1:].replace("\n","")
            pos_tagged = self.tagger.pos(lyric, True, True)
            tmp = []
            tmp.append(song[0].replace("\n",""))
            tmpstr = ""
            i = 0
            for token in pos_tagged:
                if token[1] == 'Noun' or token[1] == 'Verb' or token[1] == 'Adjective':
                #remove stopwords
                    if token not in self.stopwords:
                        tmpstr += token[0]+","

            tmpstr = tmpstr[0:len(tmpstr)-1]
            tmp.append(tmpstr)
            output.append(tmp)
            #break
        return output

    def pos4sen(self, sentence, mode = "twitter"):
        if mode == "twitter":
            self.tagger = Twitter()
        output = []
        pos_tagged = self.tagger.pos(sentence, True, True)

        for token in pos_tagged:
              if token[1] == 'Noun' or token[1] == 'Verb' or token[1] == 'Adjective':
                #remove stopwords
                if token not in self.stopwords:
                    output.append(token[0])

        return output

    def StopwordDic(self, f = ""):
        stopfile = open(f, "r", encoding="utf-8")
        for stopword in stopfile:
            self.stopwords.append(stopword)
        stopfile.close()

    def totextfile(self, doc, outputfilename):
        f = open(outputfilename, "w", encoding="utf-8")
        for line in doc:
            f.write(line)
        f.close()

def main() :
    pre = preprocess("./clustering/data/remove_lyrics.txt")
    pre.loadlyric()
    pre.StopwordDic("./clustering/data/stopwords_kr.txt")
    doc = pre.postagging()
    doc2 = []
    for docu in doc:
        str = docu[0]+";"+docu[1]+"\n"
        doc2.append(str)
    pre.totextfile(doc2, "./clustering/data/preprocessed_lyric222.txt")

if __name__ == '__main__':
    main()

