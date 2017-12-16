from LDA import *

def main():
    lda = LDA()
    topic_n = 10 # 토픽 개수
    lda.makemodel(topic_n)

    lda.load()

    n_words = 30 # 토픽 당 출력할 단어 개수
    k = 10 # 출력할 토픽 개수
    print(lda.model.show_topics(k, n_words))

if __name__ == '__main__':
    main()
