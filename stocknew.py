
import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('Stock Headlines.csv', encoding = 'ISO-8859-1')
df.dropna(inplace=True)
# print(df.shape)
df_copy = df.copy()
df_copy.reset_index(inplace=True)

train = df_copy[df_copy['Date'] < '20150101']
test = df_copy[df_copy['Date'] > '20141231']

y_train = train['Label']
train = train.iloc[:, 3:28]
y_test = test['Label']
test = test.iloc[:, 3:28]

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

train.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)
test.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)

new_columns = [str(i) for i in range(0,25)]
train.columns = new_columns
test.columns = new_columns

for i in new_columns:
  train[i] = train[i].str.lower()
  test[i] = test[i].str.lower()

train_headlines = []
test_headlines = []

for row in range(0, train.shape[0]):
  train_headlines.append(' '.join(str(x) for x in train.iloc[row, 0:25]))

for row in range(0, test.shape[0]):
  test_headlines.append(' '.join(str(x) for x in test.iloc[row, 0:25]))

ps = PorterStemmer()
train_corpus = []

for i in range(0, len(train_headlines)):
  
  words = train_headlines[i].split()

  words = [word for word in words if word not in set(stopwords.words('english'))]

  words = [ps.stem(word) for word in words]

  headline = ' '.join(words)

  train_corpus.append(headline)

test_corpus = []

for i in range(0, len(test_headlines)):
  
  words = test_headlines[i].split()

  words = [word for word in words if word not in set(stopwords.words('english'))]

  words = [ps.stem(word) for word in words]

  headline = ' '.join(words)

  test_corpus.append(headline)

down_words = []
for i in list(y_train[y_train==0].index):
  down_words.append(train_corpus[i])

up_words = []
for i in list(y_train[y_train==1].index):
  up_words.append(train_corpus[i])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000, ngram_range=(2,2))
X_train = cv.fit_transform(train_corpus).toarray()

X_test = cv.transform(test_corpus).toarray()



# from sklearn.linear_model import LogisticRegression
# lr_classifier = LogisticRegression()
# lr_classifier.fit(X_train, y_train)
# lr_y_pred = lr_classifier.predict(X_test)


from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

nb_y_pred = nb_classifier.predict(X_test)

pickle.dump(cv, open('cv-transform2.pkl', 'wb'))
filename = 'stock-model1.pkl'
pickle.dump(nb_classifier, open(filename, 'wb'))



from sklearn.metrics import accuracy_score, precision_score, recall_score
# score1 = accuracy_score(y_test, lr_y_pred)
# score2 = precision_score(y_test, lr_y_pred)
# score3 = recall_score(y_test, lr_y_pred)
# print("---- Scores ----")
# print("Accuracy score is: {}%".format(round(score1*100,2)))
# print("Precision score is: {}".format(round(score2,2)))
# print("Recall score is: {}".format(round(score3,2)))


# import re

# def stock_prediction(sample_news):
#   sample_news = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_news)
#   sample_news = sample_news.lower()
#   sample_news_words = sample_news.split()
#   sample_news_words = [word for word in sample_news_words if not word in set(stopwords.words('english'))]
#   ps = PorterStemmer()
#   final_news = [ps.stem(word) for word in sample_news_words]
#   final_news = ' '.join(final_news)

#   temp = cv.transform([final_news]).toarray()
#   return lr_classifier.predict(temp)


# from random import randint

# sample_test = df_copy[df_copy['Date'] > '20141231']
# sample_test.reset_index(inplace=True)
# sample_test = sample_test['Top1']

# row = randint(0,sample_test.shape[0]-1)
# sample_news = sample_test[row]

# print('News: {}'.format(sample_news))
# if stock_prediction(sample_news):
#   print('Prediction: The stock price will remain the same or will go down.')
# else:
#   print('Prediction: The stock price will go up!')