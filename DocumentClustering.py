import pickle
import pandas as pd
from textPreprocessing import preprocessing
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from bokeh.io import output_file
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure,show, output_notebook
from bokeh.charts import Scatter
from pprint import pprint
import numpy as np

# load the entire data_set
df = pickle.load(open("business_category_reviews.p","rb"))

# get separated reviews and categories
all_reviews = []
category = []

for idx, reviews in enumerate(df.Review.values):
    for review in reviews:
        category.append(df.Categories.values[idx])
        all_reviews.append(review)

review_dataFrame = pd.DataFrame(columns=['reviews','category'])
review_dataFrame['reviews'] = all_reviews
review_dataFrame['category'] = category


totalvocab_stemmed = []
totalvocab_tokenized = []

for i in review_dataFrame.reviews.values:
    allwords_stemmed = preprocessing(i)
    totalvocab_stemmed.extend(allwords_stemmed)

# creating vocab
vocab_frame = pd.DataFrame({'words': totalvocab_stemmed}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=200000,
                                 min_df=0.08, stop_words='english',
                                 use_idf=True, tokenizer=preprocessing, ngram_range=(1,2))

tfidf_matrix = tfidf_vectorizer.fit_transform(review_dataFrame.reviews) #fit the vectorizer to synopses

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
#print(terms)

###SVD
svd = TruncatedSVD(n_components=2,random_state=0)
svd_tfidf = svd.fit_transform(tfidf_matrix)

#### TSNE
#tsne_model = TSNE(n_components=2,verbose=1, random_state=0, n_iter=300)
#tsne_tfidf = svd.fit_transform(svd_tfidf)

print(svd_tfidf.shape)
#print(tsne_tfidf.shape)

tsne_tfidf = svd_tfidf
tsne_model = svd

#### bokeh

#output_notebook()

output_file("tfidfplot.html")
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="tfidf clustering of reviews",
                       tools="pan, wheel_zoom, box_zoom, reset, previewsave",
                       x_axis_type=None, y_axis_type=None, min_border=1)

tfidf_dtFrame = pd.DataFrame(tsne_tfidf, columns=['x','y'])
tfidf_dtFrame['reviews'] = review_dataFrame.reviews
tfidf_dtFrame['category'] = review_dataFrame.category

print(tfidf_dtFrame.columns)

plot_tfidf.scatter(source=tfidf_dtFrame, x="x", y="y")
#hover = plot_tfidf.select(dict(type=HoverTool))
#hover.tooltips = {'reviews':"@reviews","category":"@category"}
show(plot_tfidf)
#output_notebook()

print("TFIDF Done")


dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 10
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

# joblib.dump(km,  'doc_cluster.pkl')
# km = joblib.load('doc_cluster.pkl')

clusters = km.labels_.tolist()

dict_reviews = {'categories':category, 'review':all_reviews, 'cluster':clusters}
frame = pd.DataFrame(dict_reviews, index=[clusters], columns=['categories','review','cluster'])
print(frame['cluster'].value_counts())

print("Top terms per cluster:")

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()

tsne_kmeans = tsne_model.fit_transform(dist)
#print(tsne_kmeans)
output_file("plot_lsa_new.html")
colormap = np.array(["#6d8dca","#69de53","#723bca","#c3e14c","#c84dc9"])
plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="LSA clustering of reviews",
                       tools="pan, wheel_zoom, box_zoom, reset, previewsave",
                       x_axis_type=None, y_axis_type=None, min_border=1)

kmeans_dtFrame = pd.DataFrame(tsne_kmeans, columns=['x','y'])
kmeans_dtFrame['cluster'] = clusters
kmeans_dtFrame['reviews'] = review_dataFrame.reviews
kmeans_dtFrame['category'] = review_dataFrame.category

#plot_kmeans.scatter(source=kmeans_dtFrame, x="x", y="y",fill_color=colormap)

plot_kmeans = Scatter(kmeans_dtFrame, x='x',y='y', legend="top_left",color='cluster',marker='cluster')
#hover = plot_kmeans.select(dict(type=HoverTool))
#hover.tooltips = {'reviews':"@reviews","category":"@category","cluster":"@cluster"}
show(plot_kmeans)


print("Done LSA")



### DOCUMENT CLUSTERING LDA
from LDA import generate_LDA_model, get_review_text,create_document_term_matrix, display_topics

# texts = get_review_text("preprocessed_reviews.p")
# doc_term_mat, dictionary = create_document_term_matrix(texts)
# topics = generate_LDA_model(doc_term_mat, dictionary)
# display_topics(topics)


import lda
from sklearn.feature_extraction.text import CountVectorizer
cvect = CountVectorizer(min_df=2, max_features=200000,tokenizer=preprocessing,ngram_range=(1,2))
cvz = cvect.fit_transform(review_dataFrame.reviews)

lda_model = lda.LDA (n_topics=10, n_iter=50)
X_topics = lda_model.fit_transform(cvz)

print(X_topics)

n_top_words = 10
topic_summaries = []

topic_word = lda_model.topic_word_
vocab = cvect.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i," ".join(topic_words)))


#tsne_model = TSNE(n_components=2,verbose=1,random_state=0,n_iter=300)
tsne_lda = tsne_model.fit_transform(X_topics)

doc_topic = lda_model.doc_topic_
lda_keys = []

for i, review in enumerate(review_dataFrame.reviews):
    lda_keys+=[doc_topic[i].argmax()]

output_file("plot_lda.html")
plot_lda = bp.figure(plot_width=700, plot_height=600, title="LDA clustering of reviews",
                       tools="pan, wheel_zoom, box_zoom, reset, previewsave",
                       x_axis_type=None, y_axis_type=None, min_border=1)


lda_dtFrame = pd.DataFrame(tsne_lda, columns=['x','y'])
#print(lda_dtFrame)
lda_dtFrame['reviews'] = review_dataFrame.reviews
lda_dtFrame['category'] = review_dataFrame.category
lda_dtFrame['topic'] = lda_keys
#lda_dtFrame['topic'] = lda_dtFrame['topic'].map(int)


#colormap = np.array(["#6d8dca","#69de53","#723bca","#c3e14c","#c84dc9"])
#plot_lda.scatter(source=lda_dtFrame, x='x',y='y')
plot_lda = Scatter(lda_dtFrame,x='x',y='y',marker='topic',color='topic',legend="top_left")
#hover = plot_lda.select(dict(type=HoverTool))
#hover.tooltips = {"reviews":"@reviews","topic":"@topic","category":"@category"}
show(plot_lda)

print("Done LDA")