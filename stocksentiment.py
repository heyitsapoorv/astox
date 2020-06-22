import pandas as pd
import pickle
df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")
df.head()

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
# data.head(5)

for index in new_Index:
    data[index]=data[index].str.lower()
# data.head(1)

headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier




## implement BAG OF WORDS
cv=CountVectorizer(ngram_range=(2,2))
traindataset=cv.fit_transform(headlines)

# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])

# Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = cv.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)




# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-trans.pkl', 'wb'))


# Model Building

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.2)
classifier.fit(traindataset,train['Label'])

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'stock-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)