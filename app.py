from flask import Flask, render_template, request
from pickle import load
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# print(flask.__version__)

app = Flask(__name__)

# load the model
model = load(open('spam_classifier.pkl', 'rb'))
tfidf = load(open('tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    predicted_price = []
    if request.method == 'POST':
        message = request.form.get("message")

        # Extracts main points from the text data but meaning might be changed due to removal of suffix or prefix
        # to make base of each word. i.e. cleaning would be converted to clean after applying stemming.
        lemmatizer = WordNetLemmatizer()
        corpus = []

        ## Stemmer technique
        words = re.sub('[^a-zA-Z]', ' ', message)
        words = words.lower()
        words = words.split()
        print('Words', words)

        # Applying stemming on each words after removing words
        # which does not add values to the data by using stopwords() which is available in various languages.
        words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        words = ' '.join(words)
        corpus.append(words)

        print('corpus', corpus)

        # instead of YfidfVectorizer we can use CountVectorizer(Bag of Words) which counts number of occurances for each word.
        X_test = tfidf.transform(corpus).toarray()
        print('tfidf', X_test)

        my_prediction = model.predict(X_test)
        print('prediction ', my_prediction)
    return render_template('predict.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)