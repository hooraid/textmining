from LDA import *
lda = LDA()


# make model (run only first try)
#lda.makemodel()
lda.load()
#lda.saveSim()

# # load model
from token2vec import *
from LDAresult import *


# # get similarity
t2v = token2vec()
matrix = lda.sen2bow('''네 생각으로 하루를 열고
네 생각으로 텅 빈 하루를 채우고
내 생각보다 훨씬 커져버린
그리움은 원망이 되기도 해
너에게 늘 나는 부족한 것 같아
맘처럼 좁혀지지 않던 거리
그만큼 간절해져서 쉽게 상처받고
가끔씩은 너무 밉기도 해
근데 너를 보면 다시 또 사랑에 빠져
얼었던 마음은 녹아 내리고
너로 가득 채워
어쩌면 다 날 위한 연극 같아서
끝이 아닌 것 같아서
너 없는 시간들도 너와 함께였어
네 맘과는 다른 나를 보던
너와 조금씩 멀어져만 갔던 거리
그만큼 낯설어져서 쉽게 상처받고
가끔씩은 너무 밉기도 해
근데 너를 보면 다시 또 사랑에 빠져
얼었던 마음은 녹아 내리고
너로 가득 채워
어쩌면 다 날 위한 연극 같아서
끝이 아닌 것 같아서
너 없는 시간들도 너와 함께였어
세상의 모든 이별을
매일 겪는 것 같아
얼마나 더 많은 시간이 가야
괜찮아질까
근데 너를 보면 힘들던 날은 다 잊어
아팠던 맘은 다 지워 버리고
너로 다시 채워
어쩌면 다 날 위한 연극 같아서
끝이 아닌 것 같아서
너 없는 시간들도 너와 함께였어
끝이 아닌 것만 같아서''')

document_topics, word_topic, word_phi = lda.getTopic(matrix)

data = {}
data['123'] = document_topics

print(data['123'])



# dic = t2v.saveDic("./tmp/query_dic.dict", matrix)
# tf = t2v.calculateTF(matrix)
# corp = t2v.createBoW(tf, dic)
#
# query_lda = lda.getTopic(corp)
# print(query_lda)
#
# sims = lda.getSimLyric(corp)
# print(sims[0][1])
