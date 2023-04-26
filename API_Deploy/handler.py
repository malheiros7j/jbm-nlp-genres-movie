from flask import Flask, request, Response, jsonify
from genrespred.GenresPred import Genres_pred
import pandas as pd
import pickle
import chardet
import os

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus   import stopwords

from nltk.stem     import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer(r'\w+')


model = pickle.load(open('model/model_lr_tuned.pkl', 'rb'))

def tokenize_and_stem(text):
    # Tokenize the text into individual words
    tokens = tokenizer.tokenize(text.lower())

    # Apply the Snowball stemmer to each word
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    return stemmed_tokens

# Initialize API
app = Flask(__name__)

@app.route('/genres_pred/predict', methods=['POST'])
def genres_predict():

    # test_string = request.data.decode('utf-8')
    test_string = request.data.decode(chardet.detect(request.data)['encoding'])


    if not isinstance(test_string, str):
    	return jsonify({'error': 'Input data must be a string'}), 400

    # Define Genres_pred function here or import it from somewhere else
    pipeline = Genres_pred()

    # Clean Text Synopsis
    text = pipeline.clean_text(test_string)

    # Transform into TFIDF
    text_vec = pipeline.create_tfidf(text)


    # Normalize the features created
    text_vec = pipeline.normalize_feature(text_vec)


    # Predict the genres
    prediction = pipeline.get_prediction(model,text_vec)

    return jsonify(prediction), 200

if __name__ == '__main__':
    port = os.environ.get('PORT',5000)
    app.run(host='0.0.0.0',port=port)
    
