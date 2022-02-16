# sentiment-analysis

Introduction

Twitter is an online social network with over 330 million active monthly users as of February 2018. It’s ranked number eight as the most popular website in the world with an average of nearly eleven million hits a day. Users on twitter create short messages called tweets to be shared with other twitter users who interact by retweeting and responding. Twitter employs a message size restriction of 280 characters or less which forces the users to stay focused on the message they wish to disseminate. This very characteristic makes messages on twitter very good candidates for the Machine Learning (ML) task of sentiment analysis. Importance of sentiment analysis has arisen in order to solve this complex problem especially within the area of natural language processing (NLP) where, it can be defined as the computational treatment of opinions, feelings and subjectivity in texts. opinion mining find out the opinions and experience of other people over the internet using digital social media network like Facebook , reviews, forums, blogs, Twitter.

The objective of this project is writing a text classification pipeline to classify the tweets as either positive, negative or neutral and introduce at least three features besides the document vector and show how each effects the results. Accordingly, we’ve used different classifiers such as Naïve Bayes, Support Vector Machine, & Random Forest looking for the one that gives us the best accuracy along with different document vector like count vectorizer, TFIDF vectorizer, TFIDF with N-grams 1,2. Moreover, we’ve added and removed a lot of features in our way to reach the best possible accuracy.



















Data Description

The provided train data are as following twitter-2013train.txt, twitter-2015train.txt, & twitter-2016.txt. In this project, we’re given the mentioned training data that consist of:-
·	ID: identifies each tweet across the whole given data.
·	Label: indicates if the tweet was positive or negative or even natural.
·	Text: holds the tweet text.

We’ve used the whole given training data to train our classifiers and used 10 fold cross validation when reporting our results as required.

Below you can find all the features we’ve gone through which we’ll discuss each one impacts later on:-
·	Tweet length
·	Positive lexicons
·	Negative lexicons
·	Polarity of positive and negative words
·	Exclamation mark counts
·	Hash counts
·	URL counts
·	Positive emojis
·	Negative emojis

Most used words (Negative/Neutral/Positive):

NEUTRAL WORDS
POSITIVE WORDS
NEGATIVE WORDS 









Top frequent Hashtags












TOP 10 NEUTRAL HASHTAGS






TOP 10 NEGATIVE HASHTAGS








TOP 10 POSITIVE HASHTAGS



Project Experiments

·	Pre-processing
In the preprocessing phase, several things were tried:
1.	Tokenization
2.	Stopwords removal
3.	Casefolding
4.	Stemming


·	Baseline experiments

Various classifiers with various vectorization options were tried as a baseline benchmark. All of assessments were made using 10-fold cross validation. Below is the summary of the best baseline experiments in terms of results.

Baseline Experiment 1:
	Vectorizer: TfIdf 
	Algorithm: Linear SVC
Results (Accuracy 10-fold CV):

array([0.48224299, 0.52149533, 0.53894081, 0.50093458, 0.45511222,
       		0.52057357, 0.39650873, 0.37616968, 0.41422333,0.42545228])

Baseline Experiment 2:
	Vectorizer: TfIdf
	Algorithm: Naïve Bayes Multinomial()
Results (Accuracy 10-fold CV):

array([0.60685358, 0.59875389, 0.5470405 , 0.4741433 , 0.44014963,
       	0.46072319, 0.39713217, 0.18402994, 0.34185901, 0.46350593])

Baseline Experiment 3** (Baseline Best Experiment):
	Vectorizer: CountVectorizer
	Algorithm: Logistic Regression
Results (Accuracy & F1 10-fold CV):

'test_f1_macro': array([0.60815738, 0.65067963, 0.61010303, 0.51599494, 0.48309074,
        0.49769816, 0.44911722, 0.19915748, 0.3168594 , 0.39749611]),
 'test_accuracy': array([0.66417445, 0.68847352, 0.65046729, 0.56074766, 0.51246883,
        0.55112219, 0.47755611, 0.20212102, 0.32127261, 0.44291953]),


·	Enhancement experiments

Several enhancements were made on the baseline experiment to enhance the accuracy and F-score of the model
Those enhancements were as follows:
1.	Adding features:

➢	('numCaps', count number of capital letters)
➢	('countNeg', count number of negations)
➢	('length', tweet length in letters)
➢	('hashCount', number of hash-tags in tweet)
➢	('urlCount', number of URLs in tweet)
➢	('Polarity', LexiconRatioSentiment)
➢	('ExclamationMarks', number of ! marks in tweet)
➢	('PosEmoji', number of positive emojis)
➢	('NegEmoji', number of positive emojis)
➢	('numSentences', number of sentences in tweet)
➢	('avgWords', average number of words/sentence in tweet)
➢	('maxWordLength', maximum word length in tweet)
➢	('medWordLength', median of word length in tweet)

2.	Using ngram_range(1,2) and (1,3)
➢	Result: ngram_range(1,2) performed better

Results Accuracy for 10-Fold CV:
array([0.60747664, 0.64361371, 0.62679128, 0.60311526, 0.5436409 ,
       0.58665835, 0.55112219, 0.33936369, 0.39114161,0.49844042])

3.	Using class_weight = ‘balanced’
➢	Result: reached a better performance

Results Accuracy for 10-Fold CV:

array([0.62492212, 0.63613707, 0.61744548, 0.62616822, 0.56483791,
       0.60972569, 0.56359102, 0.31503431, 0.35121647,0.46849657])

4.	Using max_iter = 10000
➢	Result: reached better performance

Results Accuracy for 10-Fold CV:

array([0.6517134 , 0.66791277, 0.64361371, 0.60872274, 0.58167082,
       0.59226933, 0.58416459, 0.28883344, 0.3518403 ,0.46662508])

5.	Using word2vec
Count of id	Column Labels		
Row Labels	0	1	2
0	800	375	199
1	308	780	174
2	166	94	200
➢	Result: No major improvement



6.	Using VotingClassifier with ensemble(Logitic Regression, RandomForest, NaïveBayes)
➢	Result: Logistic Regression alone has better performance

NOTE: Boosting and gridsearch training time was huge, so no results reported in those experiments


·	Overall conclusion
o	Final Experiment:

➢	Algorithm: Logistic Regression(max_iter=10000, class_weight='balanced')
➢	CountVectorizer(n_gram_range(1,2))
➢	Stop_words removed


Confusion Matrix:
Count of id	Column Labels		
Row Labels	0	1	2
0	1105	244	25
1	415	815	32
2	268	66	126

