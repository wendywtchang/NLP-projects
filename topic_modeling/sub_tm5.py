import numpy as np
import pandas as pd
import re, nltk, gensim #spacy
# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
#import matplotlib.pyplot as plt
#matplotlib inline
# nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
nltk.download('wordnet')

# open title data
data_path = "dataset/subset/"
df_corona = pd.read_json(data_path + 'corona_topic5.json')

# Write a function to perform lemmatize and stem preprocessing steps on the data set.
stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v')) # lemmatize, pos
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# pandas.Series.map() --> pandas.core.series.Series
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html
processed_docs = df_corona["norm_tiabs"].map(preprocess)
print("Type of processed_docs: ", type(processed_docs))

pd_list = []

for sent in processed_docs:
    #print(' '.join(sent))
    pd_list.append(' '.join(sent))

# Create the Document-Word matrix
vectorizer = CountVectorizer(analyzer='word',
                             min_df=10,                        # minimum reqd occurences of a word
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(pd_list)

# Check the Sparsicity
data_dense = data_vectorized.todense()
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

# GridSearch the best LDA model
# Define Search Param
search_params = {'n_components': [3, 4, 5, 6], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)

# See the best topic model and its parameters
# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

# See the dominant topic in each document
# Document - Topics distribution
data = list(df_corona["norm_tiabs"])
best_lda_model = best_lda_model
#best_lda_model = lda_model

# Create Document - Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)


# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

df_document_topic_title = df_document_topic
df_document_topic_title['title'] = data ##
df_document_topic_title = df_document_topic.head(15)


# Visualize the LDA model with pyLDAvis
#pyLDAvis.enable_notebook()
# This is apply on sklearn best lda model (select from gesim result), topic = 7
panel = pyLDAvis.sklearn.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(panel, 'lda_corona_topic5.html')
