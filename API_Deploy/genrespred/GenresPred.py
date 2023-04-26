import pickle
import inflection
import pandas as pd
import numpy as np
import math 
import datetime
import unicodedata
import nltk
import re
import json

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus   import stopwords

from nltk.stem     import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

stop_words = set(stopwords.words('english'))

stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer(r'\w+')


class Genres_pred(object):

    def __init__(self):

        self.home_path = ''

        # fitado
        self.encoding_tfidf_vectorizer          = pickle.load(open(self.home_path + 'transformations/tfidf_transformation.pkl','rb'))

        # sem fit
        self.encoding_normalizer_transform      = pickle.load(open(self.home_path + 'transformations/norm_transformation.pkl','rb'))

        # fitado
        self.encoding_multilabel_transform      = pickle.load(open(self.home_path + 'transformations/multilabel_transformation.pkl','rb'))

    def tokenize_and_stem(text):
        # Tokenize the text into individual words
        tokens = tokenizer.tokenize(text.lower())

        # Apply the Snowball stemmer to each word
        stemmed_tokens = [stemmer.stem(word) for word in tokens]

        return stemmed_tokens

    def remove_stopwords(self,text):
        no_stopword_text = [w for w in text.split() if not w in stop_words]
        return ' '.join(no_stopword_text)

    def cont_to_exp(self,text):

        with open('transformations/contractions.txt') as file:
            data = file.read()
        contractions = json.loads(data)

        # print(contractions)

        if type(text) is str:
            for key in contractions:
                value=contractions[key]
                text=text.replace(key,value)
            return text
        else:
            return text

    def remove_accented_chars(self,text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def clean_text(self,text):
        # Remove Url
        text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)
        
        #Remove tudo entre tag <ref }
        text = re.sub(r'<ref.*?}}', '', text)
        
        # lower case
        text = text.lower()
        
        # Remove Contraction and transform into full word
        text = self.cont_to_exp(text)
        
        # Remove Special Chars or punctuation
        text = re.sub('[^A-Z a-z 0-9-]+', '',text)
        
        # Removed Accented Chars
        text = self.remove_accented_chars(text)
        
        # Remove Stopwords
        text = self.remove_stopwords(text)
        
        # Remove all non alphabeticall
        text = re.sub('[^a-zA-Z]',' ',text)
        
        # Removed duplicated spaces
        text = " ".join(text.split())
        
        # Remove numbers in form of text
        text = re.sub(r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b', '', text)
        
        return text

    def create_tfidf(self,text):
        return self.encoding_tfidf_vectorizer.transform([text])
  

    def normalize_feature(self,text):
        return self.encoding_normalizer_transform.fit_transform(text)


    def get_prediction(self,model,text):
        text_pred = model.predict(text)
        return self.encoding_multilabel_transform.inverse_transform(text_pred)

