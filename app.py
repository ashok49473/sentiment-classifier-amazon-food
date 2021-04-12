###################### Import modules #######################################
from flask import Flask, render_template, request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import re
import string
import pickle
###########################################################
total_stopwords = set(stopwords.words('english'))

negative_stop_words = set(word for word in total_stopwords if "n't" in word or 'no' in word)
# subtract negative stop words like no, not, don't etc.. from total_stopwords
final_stopwords = total_stopwords.symmetric_difference(negative_stop_words)
stemmer = PorterStemmer()

def preprocessor(review):
    # remove html tags
    HTMLTAGS = re.compile('<.*?>')
    review = HTMLTAGS.sub(r'', review)
    # remove puncutuation
    table = str.maketrans(dict.fromkeys(string.punctuation))
    review = review.translate(table)
    # remove digits
    remove_digits = str.maketrans('', '', string.digits)
    review = review.translate(remove_digits)
    # lower case all letters
    review = review.lower()
    # replace multiple white spaces with single space
    MULTIPLE_WHITESPACE = re.compile(r"\s+")
    review = MULTIPLE_WHITESPACE.sub(" ", review).strip()
    # remove stop words
    review = [word for word in review.split() if word not in final_stopwords]
    # stemming
    review = ' '.join([stemmer.stem(word) for word in review])
    return review

def predict_sentiment(text):
    review = preprocessor(text)
    x = vectorizer.transform([review])
    y = model.predict(x)[0]

    return 'positive' if y==1 else 'negative'
##########################################################
with open("tfidf_vectorizer.pkl", "rb") as f:
	vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

############################################################
app = Flask(__name__)

######################################################
@app.route("/", methods=['GET'])
def home():
   return render_template("home.html")

#######################################################
@app.route("/result", methods = ["GET","POST"])
def predict():
	if request.method == 'POST':
		review = request.form.get('review')
		sentiment = predict_sentiment(review)
		return render_template("result.html", sentiment = sentiment, flag=1)
	else:
		return render_template("home.html")

#######################################################
if __name__ == '__main__':
	app.run(debug=True)
