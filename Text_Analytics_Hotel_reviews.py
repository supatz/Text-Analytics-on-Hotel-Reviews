
# coding: utf-8

# ## Customer sentiment analysis- Hotel reviews

# In[2]:


## Import necessary packages
import bz2
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import sklearn.feature_extraction.text as text
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import words
import nltk
from string import printable
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk


# In[4]:


## Converting into pandas df
df = pd.read_csv('Hotel_Reviews.csv')
df.head()


# In[10]:


df['text']=df['text'].astype('str')


# ### Cleaning the data

# In[11]:


## Defining empty list
ReviewText_1=[]


# In[12]:


## Removing unwanted/special characters
for i in df['text']:
    i = re.sub('[^A-Za-z0-9 .:-]+', '', i)
    i = re.sub(' +',' ',i)
    ReviewText_1.append(i)


# In[13]:


df['text']=ReviewText_1


# In[14]:


df.head()


# In[15]:


df.dtypes


# In[17]:


## Filtering based on ratings
positive_lis=df.loc[df['rating'] >= 3]
negative_lis=df.loc[df['rating'] < 3]


# In[20]:


#remove the non ascii code
positive_lis['text'] = positive_lis['text'].apply(lambda x: ''.join(' ' if ord(i) < 32 or ord(i) > 126 else i for i in x))
negative_lis['text'] = negative_lis['text'].apply(lambda x: ''.join(' ' if ord(i) < 32 or ord(i) > 126 else i for i in x))


# In[21]:


positive = positive_lis['rating'].count()
negative = negative_lis['rating'].count()


# In[22]:


slices_hours = [negative, positive]
activities = ['Negative sentiments', 'Positive sentiments']
colors = ['r', 'g']
plt.pie(slices_hours, labels=activities, colors=colors, startangle=90, autopct='%1.2f%%',shadow=True)
plt


# ### Positive sentiments

# In[23]:


top_rating_list = list(positive_lis['text'])
top_rating_text = ' '.join(top_rating_list)


# In[26]:


stopwords = set(STOPWORDS)
stopwords.add('room')
stopwords.add('hotel')
stopwords.add('good')
stopwords.add('great')
stopwords.add('nice')


# In[27]:


wordcloud = WordCloud(width=1000, height=500, max_words=100, stopwords=stopwords).generate(top_rating_text)
plt.imshow(wordcloud)
plt.axis('off')


# In[28]:


def topic_analsis(comments, x, y):
    """performs the Topic Analysis
    
    :param comments: pandas series of survey response phrases as strings
    :param x: number of topics as an int
    :param y: number of top words from each topic
    :return: list of topic word frequencies
    """
    comments = list(comments)
#     for i in range(len(comments)):
#         comments[i] = str(comments[i])
        
    # This step performs the vectorization,
    # tf-idf, stop word extraction, and normalization.
    # It assumes docs is a Python list,
    #with reviews as its elements.
    docs = comments
    cv = text.TfidfVectorizer(docs, stop_words='english')
    doc_term_matrix = cv.fit_transform(docs)
 
    # The tokens can be extracted as:
    vocab = cv.get_feature_names()
    print(len(vocab))

    # Next we perform the NMF with x topics
    #from sklearn import decomposition
    num_topics = x
 
    #doctopic is the W matrix
    decomp = decomposition.NMF(n_components = num_topics,
             init = 'nndsvd')
    doctopic = decomp.fit_transform(doc_term_matrix)
 
    # Now, we loop through each row of the T matrix
    # i.e. each topic,
    # and collect the top y words from each topic.
    n_top_words = y
    
#    topic_words = []
#    weights = []
    topic_word_freq = []
    for topic in decomp.components_:
        idx = np.argsort(topic)[::-1][0:n_top_words]
        topic_words = [vocab[i] for i in idx]
        weight_value = np.sort(topic)[::-1][0:n_top_words]
        weight_value = [int(round(elem*100)) for elem in weight_value]
#        topic_words.append([vocab[i] for i in idx])
#        weights.append(weight_value)
        topic_word_freq.append(sum([[topic_words[i]] * weight_value[i] for i in range(len(topic_words))], []))
    # view topics
#     for topic in topic_words:
#         print topic
#         print "\n------------"
#     for weight in weights:
#         print weight
#         print "\n------------"
#     for topic in topic_word_freq[0:3]:
#         print topic
#         print "\n------------"
    return topic_word_freq


# In[29]:


all_ratings = positive_lis['text']


# In[30]:


topic_words_list_detractor = topic_analsis(all_ratings, 10, 30)


# In[31]:


test = ' '.join(topic_words_list_detractor[3])


# In[32]:


for word_list in topic_words_list_detractor:
    cloud_string = ' '.join(word_list)
    wordcloud = WordCloud(width=1000, height=500, max_words=100, stopwords=stopwords,collocations=False).generate(cloud_string)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')


# ## Most popular topics
# 1. Excelent service
# 2. Clean bed and environment
# 3. Perfect location
# 4. Customer enjoyed the stay
# 5. Great ammenities
# 6. Friendly and courteous staff
# 

# ## N gram

# #### The above popular topic words will help us to deep dive more into the matter

# In[33]:


irregular_detr = positive_lis[(pd.notnull(positive_lis['text']))][['text']].drop_duplicates()
#print irregular.count()
#print irregular.head()
irregular_detr.head()


# In[34]:


nltk.download('stopwords')
nltk.download('wordnet')


# In[37]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#tokenizer = RegexpTokenizer(r'\w+')

pattern = r"""(?x)                   # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |(?:[+/\-@&*])         # special characters with meanings
            """

tokenizer = RegexpTokenizer(pattern)
stops = set(stopwords.words("english"))
stops.update(['hotel','location', 'service','booked','breakfast','clean','staff','com','online']) ## as per the word cloud
stops.remove('not')

def remove_stopwords(c):
  return ' '.join([word for word in tokenizer.tokenize(c.text.lower()) if word not in stops])
  #return ' '.join(list(set(tokenizer.tokenize(c.free_text_decoded)) - set(nltk.corpus.stopwords.words('english'))))
    
irregular_detr['p_openend'] = irregular_detr.apply(remove_stopwords, axis=1)


#sys.setdefaultencoding('utf8')
import sys
if sys.version[0] == '2':
    from importlib import reload
    reload (sys)
    sys.setdefaultencoding("utf-8")

#str(oet.free_text).encode('utf8')

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

#Running lemmatization on stopwords removed text
wordnet_lemmatizer = WordNetLemmatizer()
irregular_detr['lemmatized'] = irregular_detr.p_openend.map(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(y) for y in x.split(' ')]))

#Running stemming on lemmatized text
stemmer = SnowballStemmer("english")
irregular_detr['stemmed'] = irregular_detr.lemmatized.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

#irregular.head()


# In[38]:


#Starting with the CountVectorizer/TfidfTransformer approach...
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

cvec = CountVectorizer(stop_words='english', min_df=1, max_df=.50, ngram_range=(3,5))
#cvec

# Calculate all the n-grams found in all documents
from itertools import islice
cvec.fit(irregular_detr.lemmatized)
list(islice(cvec.vocabulary_.items(), 5))

# Check how many total n-grams we have
print (len(cvec.vocabulary_))
cvec_counts = cvec.transform(irregular_detr.lemmatized)
import numpy as np
import pandas as pd
occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
#counts_df.sort_values(by='occurrences', ascending=False).head(10)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights

weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(20)
#weights_df.sort_values(by='weight', ascending=False).tail(20)


# ### Negative Ratings

# In[53]:


all_ratings = negative_lis['text'].astype('str')


# In[54]:


topic_words_list_detractor = topic_analsis(all_ratings, 10, 30)


# In[55]:


test = ' '.join(topic_words_list_detractor[3])


# In[56]:


stopwords = set(STOPWORDS)
stopwords.add('room')
stopwords.add('hotel')
stopwords.add('said')
stopwords.add('told')
stopwords.add('time')
stopwords.add('called')


# In[57]:


for word_list in topic_words_list_detractor:
    cloud_string = ' '.join(word_list)
    wordcloud = WordCloud(width=1000, height=500, max_words=100, stopwords=stopwords,collocations=False).generate(cloud_string)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis('off')


# In[39]:


irregular_detr = negative_lis[(pd.notnull(negative_lis['text']))][['text']].drop_duplicates()
#print irregular.count()
#print irregular.head()
irregular_detr.head()


# In[59]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#tokenizer = RegexpTokenizer(r'\w+')

pattern = r"""(?x)                   # set flag to allow verbose regexps
              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
              |(?:[+/\-@&*])         # special characters with meanings
            """

tokenizer = RegexpTokenizer(pattern)
stops = set(stopwords.words("english"))
stops.update(['hotel','location', 'service','rooms','breakfast','clean','staff','smoking','water','desk'])
stops.remove('not')

def remove_stopwords(c):
  return ' '.join([word for word in tokenizer.tokenize(c.text.lower()) if word not in stops])
  #return ' '.join(list(set(tokenizer.tokenize(c.free_text_decoded)) - set(nltk.corpus.stopwords.words('english'))))
    
irregular_detr['p_openend'] = irregular_detr.apply(remove_stopwords, axis=1)


#sys.setdefaultencoding('utf8')
import sys
if sys.version[0] == '2':
    from importlib import reload
    reload (sys)
    sys.setdefaultencoding("utf-8")

#str(oet.free_text).encode('utf8')

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

#Running lemmatization on stopwords removed text
wordnet_lemmatizer = WordNetLemmatizer()
irregular_detr['lemmatized'] = irregular_detr.p_openend.map(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(y) for y in x.split(' ')]))

#Running stemming on lemmatized text
stemmer = SnowballStemmer("english")
irregular_detr['stemmed'] = irregular_detr.lemmatized.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))

#irregular.head()


# In[60]:


#Starting with the CountVectorizer/TfidfTransformer approach...
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

cvec = CountVectorizer(stop_words='english', min_df=1, max_df=.50, ngram_range=(3,5))
#cvec

# Calculate all the n-grams found in all documents
from itertools import islice
cvec.fit(irregular_detr.lemmatized)
list(islice(cvec.vocabulary_.items(), 5))

# Check how many total n-grams we have
print (len(cvec.vocabulary_))
cvec_counts = cvec.transform(irregular_detr.lemmatized)
import numpy as np
import pandas as pd
occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
#counts_df.sort_values(by='occurrences', ascending=False).head(10)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights

weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(20)
#weights_df.sort_values(by='weight', ascending=False).tail(20)

