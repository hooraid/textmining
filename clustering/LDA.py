from gensim import corpora, models

dictionary = corpora.Dictionary.load("./tmp/dictionary.dict") # load dictionary
corpus = corpora.MmCorpus("./tmp/corpus.mm") # load corpus

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
