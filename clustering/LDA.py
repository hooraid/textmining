from gensim import corpora, models, matutils
from preprocess import *

class LDA():
    model = None
    dictionary = None
    corpus = None

    def __init__(self, dicfile="./tmp/dictionary.dict", corfile="./tmp/corpus.mm"):
        self.dicfile = dicfile
        self.corfile = corfile
        self.dictionary = corpora.Dictionary.load(dicfile)  # load dictionary
        self.corpus = corpora.MmCorpus(corfile)  # load corpus

    def makemodel(self,top_n=100):
        self.top_n = top_n
        lda = models.LdaModel(self.corpus, id2word=self.dictionary, num_topics=top_n)
        lda.save("./tmp/lda.model")
        lda.print_topic()

    def load(self, filename="./tmp/lda.model"):
        self.model = models.LdaModel.load(filename)
        #print(self.model.print_topic(1))

    def sen2bow(self,sen):
        pre = preprocess()
        tokens = pre.pos4sen(sentence=sen)
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

        sim = matutils.cossim(matutils.dense2vec(sentence1), matutils.dense2vec(sentence2))
        return sim
