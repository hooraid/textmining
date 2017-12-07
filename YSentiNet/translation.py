import os
import requests, json
from bs4 import BeautifulSoup
#api_key = 'AIzaSyCTN9lahZfT_rFufb34cd28CnkoRX0qSgg'
#base_url = "https://translation.googleapis.com/language/translate/v2?key=AIzaSyCTN9lahZfT_rFufb34cd28CnkoRX0qSgg"
#data= {'key': 'AIzaSyCTN9lahZfT_rFufb34cd28CnkoRX0qSgg'}
#res = requests.post(base_url)


target = 'ko'

input=open('output.txt','r')
output=open('output2.txt','w')

count = 0
strlist = []


def translate(enword):
    enword = 'accueil'
    query = "http://endic.naver.com/search.nhn?sLn=kr&query="+enword+"&searchOption=entry_idiom"
    html = requests.get(query)
    soup = BeautifulSoup(html.content, 'html.parser')
    #content > div.word_num_nobor > dl > dd:nth-child(2) > div > p:nth-child(1) > span.fnt_k05
    result = soup.find("span", {"class":"fnt_k05"})
    if result==None :
        return '000'
    koword = (result.text).split(',')[0]
    return koword

for ss in input:
    count=count+1
    strlist.append(ss)

    if count!=10 :
        continue
    
    enWord = ss.split('\t')[0]
    #datas = {'key':'AIzaSyCTN9lahZfT_rFufb34cd28CnkoRX0qSgg', 'q' : enWord, 'target' : 'ko'}
    #translation = requests.post(base_url, data=json.dumps(datas) )
    #koWord = translation.json()['data']['translations'][0]['translatedText']
    koWord= translate(enWord)
    if koWord=='000' :
        
    for txt in strlist:
        output.writelines(txt.replace(enWord,koWord))
    
    count=0
    strlist.clear()


input.close()
output.close()