"""
    Code for Performing Document Clustering using LDA
"""


from gensim import corpora, models
import pickle
from pprint import pprint

"""
    Method to load the tokenized reviews file
    :parameter infile - path of the input file
    :returns pickle object i.e.
    List of Lists in our case containing Tokens generated for every review
"""
def get_review_text(infile):
    return pickle.load(open(infile,"rb"))

"""
    Method to generate a Document term Matrix for LDA
    Finds how frequently each term occurs within each document
    :parameter texts - cleaned review text tokens
    :returns doc_term - document term matrix mapping
"""
def create_document_term_matrix(texts):

    # creates a dictionary for each text in texts and assigns a unique id to each
    dictionary = corpora.Dictionary(texts)
    # print(dictionary)
    # print(dictionary.token2id['found'])
    # doc2bow function converts dictionary into bag of words. gives a tuple representation
    # termID, term frequency
    doc_term = [dictionary.doc2bow(text) for text in texts]
    # print(texts[0])
    # print(doc_term[0])
    # print()
    return doc_term, dictionary


"""
    Method to generate LDA Model
    :parameter dtm - doc-term matrix
    :returns topics
"""
def generate_LDA_model(dtm, dictionary):

    lda_model = models.ldamodel.LdaModel(corpus=dtm, num_topics=5, id2word=dictionary, passes=10)
    print(lda_model)
    topics = lda_model.show_topics(num_topics=5, num_words=10, formatted=False)

    # print(topics)
    # print_topics(topics)

    return topics

"""
    Method to print Topics found
    :parameter topics - List of topics
"""
def display_topics(topics):
    i = 1
    for topic in topics:
        print("Topic #" + str(i) + ":", end=" ")
        for elem in topic[1]:
            print(elem[0], end=" ")
        i+=1
        print("\n")


"""
    Method to generate LSA Model
"""
def generate_LSA_Model(dtm, dictionary):

    tfidf = models.TfidfModel(dtm)
    dtm_tfidf = tfidf[dtm]

    lsi = models.lsimodel.LsiModel(corpus=dtm_tfidf, id2word=dictionary, num_topics=10, power_iters=5)

    topics = lsi.show_topics(10,formatted=False)
    return topics


"""
    Main Method
"""
if __name__ == "__main__":

    # getting the reviews
    infile = "preprocessed_reviews.p"
    #infile = "preprocessed_content.p"
    texts = get_review_text(infile)
    # print(len(texts))
    # print(texts)

    # creating a document term matrix mapping
    doc_term_mat, dictionary = create_document_term_matrix(texts)

    # Applying LDA
    topics = generate_LDA_model(doc_term_mat, dictionary)
    display_topics(topics)

    # Applying LSA
    #topics = generate_LSA_Model(doc_term_mat, dictionary)
    #display_topics(topics)

