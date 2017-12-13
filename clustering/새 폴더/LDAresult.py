from LDA import *

lda = LDA()
topic_n = 100 # 토픽 개수
lda.makemodel(topic_n)

n_words = 20 # 토픽 당 출력할 단어 개수
k = 10 # 출력할 토픽 개수
lda.model.show_topics(num_topics=k, num_words=n_words, log=False, formatted=True)

if __name__ == '__main__':
    main()
