# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import re

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'stock-model1.pkl'
nb_classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform2.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		sample_news =[message]
		# sample_news = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_news)
		# sample_news = sample_news.lower()
		# sample_news_words = sample_news.split()
		# sample_news_words = [word for word in sample_news_words if not word in set(stopwords.words('english'))]
		# ps = PorterStemmer()
		# final_news = [ps.stem(word) for word in sample_news_words]
		# final_news = ' '.join(final_news)
		temp = cv.transform(sample_news).toarray()
		my_prediction = nb_classifier.predict(temp)

		return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)