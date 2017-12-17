from gensim import corpora, models, matutils
from preprocess import *
import numpy as np

class LDA():
    model = None
    dictionary = None
    corpus = None

    def __init__(self, dicfile="./tmp/dictionary.dict", corfile="./tmp/corpus.mm"):
        self.dicfile = dicfile
        self.corfile = corfile
        self.dictionary = corpora.Dictionary.load(dicfile)  # load dictionary
        self.corpus = corpora.MmCorpus(corfile)  # load corpus
        self.pre = preprocess()

    def makemodel(self,top_n=100):
        self.top_n = top_n
        lda = models.LdaModel(self.corpus, id2word=self.dictionary, num_topics=top_n)
        lda.save("./tmp/lda.model")
        #lda.print_to()

    def load(self, filename="./tmp/lda.model"):
        self.model = models.LdaModel.load(filename)
        #print(self.model.print_topics(num_topics=k, num_words=n_words))

    def sen2bow(self,sen, doPrepro=True):
        if doPrepro:
            tokens = self.pre.pos4sen(sentence=sen)
        else:
            tokens = sen.split(',')
        bow = self.model.id2word.doc2bow(tokens)
        return bow

    def getTopic(self, sentence):
        '''
        :param sentence: preprocessed bow type
        :return:
        '''
        #print(self.model.get_document_topics(sentence))
        return self.model.get_document_topics(sentence, per_word_topics=True)

    def getSim(self, sentence1, sentence2):
        '''
        :param sentence: preprocessed bow
        :return:
        '''
#        sim = matutils.cossim(matutils.dense2vec(sentence1), matutils.dense2vec(sentence2))
        # print()
        # print(sentence1)
        # print(self._sentovec(sentence1))
        # print(matutils.dense2vec(self._sentovec(sentence1)))
        # print('--------')
        # print(sentence2)
        # print(self._sentovec(sentence2))
        # print(matutils.dense2vec(sentence2))
        # print()
        sim = matutils.cossim(sentence1, sentence2)
        return sim

    def _sentovec(self, sen):
        output = np.zeros(10)
        for vec in sen:
            output[vec[0]] = vec[1]
        return output