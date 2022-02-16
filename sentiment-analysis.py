#!/usr/bin/env python
# coding: utf-8

# In[1]:


#http://localhost:8888/notebooks/Cleo%20Classifier.ipynb#
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from nltk.stem.porter import PorterStemmer
import json
from sklearn import naive_bayes
import collections
import gensim
from gensim.models import KeyedVectors
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
#!wget -P /root/input/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"


# In[ ]:


model = KeyedVectors.load_word2vec_format('D:\\GoogleNews-vectors-negative300.bin', binary=True)
word_vectors = model.wv


# In[ ]:


VectorizedData = MyVectorizer(tweets['text'])


# In[2]:


os.chdir('C:\\Users\\v14wezz\\Desktop\\Data Science\\Big Data @NU\\Practical Data Mining\\Term Project\\Training Data\\')
tweets2013 = 'twitter-2013train.txt'
tweets2015 = 'twitter-2015train.txt'
tweets2016 = 'twitter-2016train.txt'

tweets_2013 = pd.read_csv(tweets2013,sep = "\t", header = None)
tweets_2015 = pd.read_csv(tweets2015,sep = "\t", header = None)
tweets_2016 = pd.read_csv(tweets2016,sep = "\t", header = None)

tweets = tweets_2013.append(tweets_2015).append(tweets_2016)
tweets.columns = ['id', 'sentiment', 'text']
len(tweets)


# In[3]:


stop_words = set(stopwords.words('english'))
def tokenize(text):
    tokens = text.split()
    stems = ""
    for item in tokens:
        if item not in stop_words:
            stems = stems +" "+ PorterStemmer().stem(item.casefold())
    return stems
#nltk.download('punkt')
corpus = [tokenize(i) for i in tweets['text']]


# In[4]:


all_tweets=tweets["text"]
#all_tweets=pd.Series(corpus)
Y=tweets['sentiment']
#vectorizer = CountVectorizer().fit(all_tweets)
#X = vectorizer.transform(all_tweets)


# In[5]:


label_dict = {'neutral' : 0, 'positive': 1, 'negative': 2}
Y.replace(label_dict,inplace=True)


# In[ ]:


def Naive_bayes_clf(tweet,label):
    clfr = naive_bayes.MultinomialNB()
    CV=cross_validate(clfr,tweet,label,cv=10,scoring=('f1_macro','accuracy'))
    return CV


# In[73]:


vect = CountVectorizer().fit(all_tweets)
X = vect.transform(all_tweets)
lr = LogisticRegression()
CV=cross_validate(lr,X,Y,cv=10,scoring=('f1_macro','accuracy'))


# In[74]:


CV


# In[ ]:


Naive_bayes_clf(X,Y)


# In[6]:


os.chdir('C:\\Users\\v14wezz\\Desktop\\Data Science\\Big Data @NU\\Practical Data Mining\\Term Project\\Potentially Useful Files\\')
pos_words = pd.read_csv("positive-words.txt", names=["pos_words"], encoding="latin1")
neg_words = pd.read_csv("negative-words.txt", names=["neg_words"], encoding="latin1")
p_list = pos_words['pos_words'].tolist()
n_list = neg_words['neg_words'].tolist()


# In[7]:


class NegativeEmoji(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def getNegEmoji(self, name):
        """Helper code to compute average word length of a name"""
        n_emoji = [":(" , ':-(' , ":((" , ":'(", ":-(", ":(", "(:", "(-:", ":,(", ":'(",  ":(("]
        return sum(el in n_emoji for el in name)

      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.getNegEmoji).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[8]:


class PositiveEmoji(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def getPosEmoji(self, name):
        """Helper code to compute average word length of a name"""
        p_emoji = [":)" , ":-)" , ":))" , ":D", ";)", "(-:", "(:", ":-D", ":D", "X-D", "XD", "xD", "<3", ":*", ";-)", ";)", ";-D", ";D", "(;", "(-;"]
        return sum(el in p_emoji for el in name)

      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.getPosEmoji).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[9]:


class TweetLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def tweet_length(self, name):
        """Helper code to compute average word length of a name"""
        return len(name)
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.tweet_length).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[10]:


class LexiconPosSentiment(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def countPos(self, name):
        """Helper code to compute average word length of a name"""
        return sum([1 for word in name.split() if word in p_list])
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.countPos).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[11]:


class LexiconNegSentiment(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def countNeg(self, name):
        """Helper code to compute average word length of a name"""
        return sum([1 for word in name.split() if word in n_list])
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.countNeg).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[12]:


class LexiconRatioSentiment(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def calcRatio(self, name):
        """Helper code to compute average word length of a name"""
        pos = sum([1 for word in name.split() if word in p_list]) 
        neg = sum([1 for word in name.split() if word in n_list])
        total =  pos + neg
        if total > 0:
            return ((pos - neg)/total) 
        else:
            return 0
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.calcRatio).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[13]:


class MarksCounter(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def marksCount(self, name):
        """Helper code to compute average word length of a name"""
        return name.count('!')
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.marksCount).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[14]:


class hashCounter(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def hashCount(self, name):
        """Helper code to compute average word length of a name"""
        return name.count('#')
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.hashCount).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[15]:


class urlCounter(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def urlCount(self, name):
        """Helper code to compute average word length of a name"""
        return name.count('http')
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.urlCount).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[16]:


class MyVectorizer(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def tweet2Vec(self,tweet):
        words = tweet.split()
        tweet2Vec = np.zeros(300,)
        count = 0
        for word in words:
            if word in word_vectors.vocab:
                tweet2Vec = np.add(tweet2Vec,model[word])
                count = count + 1
        tweet2Vec = np.divide(tweet2Vec,count)
        tweet2Vec = np.nan_to_num(tweet2Vec)
        #print(list(tweet2Vec))
        return list(tweet2Vec)
    
#   def MyVectorizer(dataset):
#      vectorized_data = np.zeros((300,))
#        for tweet in dataset:
#            tweet_vectorized, count = tweet2Vec(tweet)
#            vectorized_data = np.append(vectorized_data,tweet_vectorized,axis = 0)
#        vectorized_data = vectorized_data[300:].reshape(len(dataset),300)
#        vectorized_data = np.nan_to_num(vectorized_data)
#        return vectorized_data
    
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        #print(len(df.apply(self.tweet2Vec)[0]))
        return df.apply(self.tweet2Vec)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[17]:


class countSentences(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def urlCount(self, name):
        """Helper code to compute average word length of a name"""
        return len(name.split('.'))
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.urlCount).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[18]:


class avgWordsinSentence(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def avgWords(self, name):
        """Helper code to compute average word length of a name"""
        return np.mean([len(i.split()) for i in name.split('.')])
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.avgWords).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[67]:


class medianWordsLength(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def medWords(self, name):
        """Helper code to compute average word length of a name"""
        return np.median([len(i) for i in name.split()])
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.medWords).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[19]:


class maxWordLength(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def maxWord(self, name):
        """Helper code to compute average word length of a name"""
        return np.max([len(i) for i in name.split()])
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.maxWord).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[20]:


class countCapitals(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def countCapital(self, name):
        """Helper code to compute average word length of a name"""
        cap = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        return sum(el in cap for el in name.split())
      
    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.countCapital).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[54]:


class countNegations(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def countNeg(self, name):
        """Helper code to compute average word length of a name"""
        text = name.replace("'", "").lower()
        negations = ['never','no','nothing','nowhere','noone','none','not','havent','hasnt','hadnt','cant','couldnt','shouldnt','wont','wouldnt','dont','doesnt','didnt','isnt','arent','aint']
        return sum(elem in text.split()  for elem in negations)

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df.apply(self.countNeg).values.reshape(-1,1)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


# In[68]:


#X_train, X_test, Y_train, Y_test = train_test_split(all_tweets, Y, test_size=0.2, random_state=0)
#param_grid = {'max_depth': [50,  100, None], 'n_estimators':[300, 1000], 'class_weight':[{0:1, 1:1, 2:4},{0:1, 1:1, 2:3}]}

pipeline = Pipeline([ ('feats', FeatureUnion([ ('vect',CountVectorizer(stop_words=stop_words,ngram_range=(1,2))),
                                                                        ('numCaps', countCapitals()),
                                                                        ('countNeg', countNegations()),
                                                                        ('length', TweetLengthExtractor()),
                                                                        ('hashCount', hashCounter()),
                                                                        ('urlCount', urlCounter()),
                                                                        ('Polarity', LexiconRatioSentiment()),
                                                                        ('ExclamationMarks', MarksCounter()),
                                                                        ('PosEmoji', PositiveEmoji()),
                                                                        ('NegEmoji', NegativeEmoji()),
                                                                        ('numSentences', countSentences()),
                                                                        ('avgWords', avgWordsinSentence()),
                                                                        ('maxWordLength',maxWordLength()), 
                                                                        ('medWordLength',medianWordsLength())
                                                                     ])),
                                  ('clfr', LogisticRegression(max_iter=10000, class_weight='balanced')),                                                                        
                                ])
  
#('lr',LogisticRegression(max_iter=10000, class_weight='balanced')),
#scores = cross_val_score(pipeline, all_tweets, Y, cv=5)
#scores
#VotingClassifier([('lr',LogisticRegression(max_iter=10000, class_weight='balanced')),
#                                                                        ('rf',RandomForestClassifier(n_estimators=100, class_weight='balanced')),
#                                                                       ('svm', LinearSVC(max_iter=5000,C=0.01))
#                                                                     ],voting='hard')


# In[69]:


model_clf = pipeline.fit(all_tweets,Y)
#model_train = RandomForestClassifier(n_estimators=100, class_weight={0:1,1:20,2:50}).fit(VectorizedData.reshape(16041,300),Y)


# In[70]:


os.chdir('C:\\Users\\v14wezz\\Desktop\\Data Science\\Big Data @NU\\Practical Data Mining\\Term Project\\')
test_data = pd.read_excel('test_data_2906.xlsx')
IDs = test_data['id']
Y_Pred = model_clf.predict(test_data['tweet'])


# In[ ]:


#VectorizedTestData = MyVectorizer(test_data['tweet'])
#Y_Pred_w2v = model_train.predict(VectorizedTestData.reshape(3096,300))


# In[71]:


submission = pd.concat([pd.Series(IDs), pd.Series(Y_Pred)], axis=1)
submission.columns = ['id','label']
#submission
submission.to_csv('submission_53.txt', sep=',')

