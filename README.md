# CS522_Document_Clustering_Using_Topic_Modeling_Techniques
Document clustering or text clustering is an application of cluster analysis to textual documents. Using this clustering mechanism and its different implementations we will focus on modelling topics and clustering the documents based on these topics. The main aim of this project is to provide an overview of some widely-used document clustering techniques. We will compare three different techniques viz. Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA) and Word2Vec and, analyze our results to learn which technique is better. For better analysis, we will be working over 2 datasets: Enron Email Dataset and Yelp Dataset, to carry out experiment on multiple different topics.

Input Data
ENRON EMAIL DATASET
•	Enron email dataset contains data from 150 Enron users in the form of messages. The data contains a total of about 0.5 million messages which is organized into folders. The size of data on our disk is 1.78GB. This is composed by the CALO Project (A Cognitive Assistant that Learns and Organizes).
•	Source: https://www.cs.cmu.edu/~enron/
•	Data Format: The data is in Zip format and is organized into folders, one folder per user. For each user, all the messages have been stored under text files, divided into different categories like Inbox, Sent Items, etc.

YELP DATASET
•	The dataset is from the 9th round of “YELP Dataset Challenge”.Yelp Challenge Data are in JSON format, of which the size is around 4.5GB in total, containing information for 77,079 local businesses in 11 cities across 4 countries. 
•	There are two JSON files contain the information:
 yelp_academic_dataset_review.json
 yelp_academic_dataset_business.json
•	Source: https://www.yelp.com/dataset_challenge
•	Data Format: The two JSON files covers reviews of customer per business category. The Business data contains business ID and their categories. While, the reviews file contains of business ID and review ID.

Approach
The techniques we are implementing are:
•	Latent Dirichlet Allocation (LDA)
•	Latent Semantic Analysis (LSA) and
•	Word2Vec

Latent Dirichlet Allocation (LDA): 
Latent Dirichlet allocation is a way of automatically discovering topics that documents or sentences contain. It is basically a generative model and it creates a probabilistic model of how the words in each document were generated/written. LDA will determine which words are likely generated from a specific topic, then determine the topic of a document by examining these probabilities. LDA takes the Document Term Matrix as input and we do need to provide the number of topics we need to get as a parameter apriori.
 
Latent Semantic Analysis (LSA): 
It is a mathematical method that tries to bring out latent relationships within documents, by looking at all the documents and terms within them to identify relationships. Terms that are close in meaning will occur in similar pieces of text. LSA is a concept of dimensionality reduction and hence uses a mathematical technique, called Singular Value Decomposition, to a term-document matrix. SVD helps reduce the number of terms while preserving the similarity structure among documents. In SVD, a rectangular matrix is decomposed into the product of three other matrices:
a term concept vector, a concept document vector and a singular values diagonal matrix.
 
Word2Vec: 
The word2vec is used to produce word embeddings. It takes a text corpus as input and produces the word vectors as output. It first constructs a vocabulary from the training text data and then learns vector representation of words. Word Vectors are positioned in the vector space such that words that share common contexts in the corpus are near one another in the space. Word2Vec can use any of the 2 model architectures:
•	Continuous Bag-of-Words model (CBOW)
•	Skip-Gram model
In the continuous bag-of-words architecture, the model predicts the current word from a window of surrounding context words. The order of context words does not influence. In the continuous skip-gram architecture, the model uses the current word to predict the surrounding window of context words. The skip-gram architecture weighs nearby context words more heavily than more distant context words.The dataset we are performing our experimental analysis on is textual data. The aforementioned techniques will help us to get the latent topics from the textual data and using these topics we can then cluster the documents accordingly. LDA, LSA and Word2Vec are most commonly used topic modelling and clustering techniques and using them for our evaluation purpose will help us understand the data better and will provide us useful insights into the data. Also by comparing them on different datasets will help us understand which technique works better for what kind of data.  
