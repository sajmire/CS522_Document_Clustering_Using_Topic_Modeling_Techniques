import json
from pprint import pprint
from itertools import islice
from collections import defaultdict
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from wordcloud import WordCloud

"""
    Method to get a slice of Dictionary
    :parameter n - number of elements needed
    iterable - any iterable object
    :returns list of values
"""
def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


"""
    Method to read business JSON file
    :parameter infile - path of input file
    :returns dictionary with business_id as KEY and categories as VALUE
"""
def read_business_json(infile):
    business_data = {}
    with open(infile, encoding="utf8") as fp:
        for line in fp:
            entire_data = json.loads(line)
            business_data[entire_data['business_id']] = entire_data['categories']
    return business_data


"""
    Method to read review JSON file
    :parameter infile - path of input file
    n - number of lines to read
    :returns dictionary with business_id as KEY and text as VALUE
"""
def read_review_json(infile, n):
    review_data = defaultdict(list)
    read_lines = 0

    with open(infile, encoding="utf8") as fp:
        for line in fp:
            if read_lines != n:
                entire_data = json.loads(line)
                review_data[entire_data['business_id']].append(entire_data['text'])
                read_lines+=1
                #print(read_lines, len(review_data))
            else:
                break

    return review_data


"""
    Method to preprocess the data, preprocessing includes
    1. converting to lower case
    2. removing numerics, punctuations, stop words
    3. stemming

    :parameter content_line - the data to be preprocessed
    :returns tokenize words
"""
def preprocessing(content_line):

   #tolowercase
   content_line = content_line.lower()
   #print(content_line)

   # removing numerics
   words = re.sub(r'\d+', '',content_line)
   #print(words)

   #remove punctuation and split into seperate words
   words = re.findall(r'\w+', words, flags = re.UNICODE | re.LOCALE)

   # removing stop words
   stop = set(stopwords.words('english'))
   filtered_words = [word for word in words if word not in stop]
   #print("filtered words", filtered_words)

   #stemming
   stemmer = SnowballStemmer("english")
   stem_words = [stemmer.stem(w) for w in filtered_words]

   return stem_words


if __name__ == "__main__":

    # reading JSON for Business Dataset, getting Business ID and Categories
    business_infile = "yelp_academic_dataset_business.json"
    business_data_read = read_business_json(business_infile)
    print(len(business_data_read))

    #print(set([val for val in business_data_read.values()]))
    # pprint(take(5, business_data_read.items()))

    # writing the dictionary to a pickle object file
    pickle.dump(business_data_read, open("business_data.p","wb"))

    # reading JSON for review Dataset, getting Business ID and review text
    review_infile = "yelp_academic_dataset_review.json"
    review_data_read = read_review_json(review_infile, 20000)
    print(len(review_data_read.values()))
    #pprint(take(len(review_data_read), review_data_read.items()))

    # writing the dictionary to a pickle object file
    pickle.dump(review_data_read, open("review_data.p","wb"))

    # creating dataframe
    df_business = pd.DataFrame([[bus_id, categ] for bus_id, categ in business_data_read.items()], columns=['Business Id','Categories'])
    df_review = pd.DataFrame([[bus_id, review] for bus_id, review in review_data_read.items()], columns=['Business Id','Review'])

    #print(df_business)
    #print(df_review)

    # merging the Business Data and Review Data on basis of Business Id, fetching Business Id, Categories and Reviews
    final_merged_data = df_business.set_index('Business Id').join(df_review.set_index('Business Id'),how='right')

    #print(final_merged_data['Categories'])
    #pprint(final_merged_data['Review'])

    # writing the data for clustering purpose
    pickle.dump(final_merged_data, open("business_category_reviews.p","wb"))

    # here we get the preprocessed data, for every review we have
    pre_processed_reviews = []
    #with open("reviews.txt","w") as review_fp:
    for reviews in final_merged_data.Review.values:
        for review in reviews:
            #review_fp.write(review)
            pre_processed_reviews.append(preprocessing(review))


    # list of lists, tokenized words
    pickle.dump(pre_processed_reviews, open("preprocessed_reviews.p", "wb"))