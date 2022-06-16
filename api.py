import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
import gensim
import spacy
import spacy_fastlang
from collections import Counter
from scipy import sparse
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import json
import importlib_resources
from pkg_resources import resource_filename
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import joblib

def sentiment_preprocess(X_arr):
    X_arr_2 = []
    for X in X_arr:
        X = " ".join(re.sub("\n", " ", X).split())
        X = " ".join(re.sub(r"(?:\@|https?\://)\S+", " ", X).split())
        X = " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", X).split())
        X = X.lower()
        X = X.split()
        X = [word for word in X if not word in stop_words]
        with importlib_resources.open_text("taglish_sentiment", "stopwords-tl.json") as file:
            df_stop_words_tl = pd.read_json(file)
        stop_words_tl = df_stop_words_tl.iloc[:,0].tolist()
        X = [word for word in X if not word in stop_words_tl]
        X = [word for word in X if not re.match("(a)*(h)*(ha)+(a)*(h)*", word)]
        X = [word for word in X if not re.match("(e)*(h)*(he)+(e)*(h)*", word)]
        X = [word for word in X if not re.match("(a)(h)(h)*", word)]
        X = [word for word in X if not len(word)==1]
        X = [word for word in X if not word.isdigit()]
        X = " ".join(X)
        X_arr_2.append(X)
    return X_arr_2
    
def sentiment_new_features(X):
    X_pos = [nltk.pos_tag(str(tweet_words).split()) for tweet_words in X]
    X_pos_removed = [[(word, tag) for word, tag in tweet_words if tag in ('JJ', 'JJR','JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG','VBN','VBP','VBZ')] for tweet_words in X_pos]
    X_pos_only = [[tag for word, tag in tweet_words] for tweet_words in X_pos_removed]
    X_pos_counts = [Counter(tweet_words) for tweet_words in X_pos_only]
    
    df_pos = pd.DataFrame(columns = ['JJ', 'JJR','JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG','VBN','VBP','VBZ'], data = X_pos_counts)
    df_pos = df_pos.fillna(0)
    df_pos2 = pd.DataFrame()
    df_pos2["Verb"] = df_pos["VBN"] + df_pos["VB"] + df_pos["VBD"] + df_pos["VBG"] + df_pos["VBZ"] + df_pos["VBP"]
    df_pos2["Adverb"] = df_pos["RB"] + df_pos["RBR"] + df_pos["RBS"]
    df_pos2["Noun"] = df_pos["NN"] + df_pos["NNS"] + df_pos["NNP"]
    df_pos2["Adjective"] = df_pos["JJ"] + df_pos["JJS"] + df_pos["JJR"]
    df_pos = df_pos2
    df_pos = df_pos.fillna(0)

    df_adjectives = pd.DataFrame()
    arr_positive = []
    arr_comparative = []
    arr_superlative = []
    for tweet in X_pos:
        positive = 0
        comparative = 0
        superlative = 0
        for i in range(0, len(tweet)):
            word = tweet[i]
            if word[1]=="JJ" or word[1]=="JJS" or word[1]=="JJR":
                adj = word[0]
                if adj.endswith("er") or (i!=0 and tweet[i-1]=="more") or (i!=0 and tweet[i-1]=="mas"):
                    comparative = comparative + 1
                elif adj.endswith("est") or (i!=0 and tweet[i-1]=="most") or (i!=0 and tweet[i-1]=="pinaka") or adj.startswith("pinaka"):
                    superlative = superlative + 1
                else:
                    positive = positive + 1
        arr_positive.append(positive)
        arr_comparative.append(comparative)
        arr_superlative.append(superlative)
    df_adjectives["Positive (Adj)"] = arr_positive
    df_adjectives["Comparative (Adj)"] = arr_comparative
    df_adjectives["Superlative (Adj)"] = arr_superlative

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("language_detector")
    X_lang = [[nlp(word)._.language for word in (str(tweet_words).split())] for tweet_words in X]

    affixes = ["ma","na","mag","pag","nag","pala","mala","pang","magka", "nagka", "pagka","pina"] #prefixes
    df_has_affix = pd.DataFrame()
    for affix in affixes:
        df_has_affix[affix] = [[True if word.startswith(affix) else False for word in (str(tweet_words).split())] for tweet_words in X]
    df_has_affix["all"] = df_has_affix[affixes[0]]
    for affix in affixes:
        df_has_affix["all"] = [np.logical_or(df_has_affix["all"][i], df_has_affix[affix][i]) for i in range(0, len(df_has_affix))]
    df_has_affix = df_has_affix["all"]
    df_has_affix
    X_lang2 = X_lang.copy()
    for i in range(0, len(df_has_affix)):
        arr_lang = X_lang2[i]
        arr_affix = df_has_affix[i]
        X_lang2[i] = ["tl" if (arr_affix[i]==True) else "en" if arr_lang[i]=="en" else "tl" if arr_lang[i]=="tl" else "other" for i in range(0, len(arr_lang))]

    X_lang_joined = [" ".join(tweet_words) for tweet_words in X_lang2]
    X_cs_1 = [tweet_words.count("en tl") for tweet_words in X_lang_joined]
    X_cs_2 = [tweet_words.count("tl en") for tweet_words in X_lang_joined]
    X_cs_words = [tweet_words.count(" ") for tweet_words in X_lang_joined]
    X_cs = np.add(X_cs_1, X_cs_2)
    df_cs = pd.DataFrame()
    df_cs["CS_freq"] = X_cs
    df_cs = df_cs.fillna(0)
    
    df_lang_arr = pd.DataFrame(data=X_lang2)
    X_lang_counts = [Counter(tweet_words) for tweet_words in X_lang2]
    df_lang = pd.DataFrame(columns=["en","tl","other"], data=X_lang_counts)
    df_lang = df_lang[["en","tl","other"]]
    df_lang = df_lang.fillna(0)
    df_lang['first_word_en'] = np.where(df_lang_arr.iloc[:, 0]=="en", 1, 0)
    df_lang['first_word_tl'] = np.where(df_lang_arr.iloc[:, 0]=="tl", 1, 0)
    df_lang['first_word_other'] = np.where(df_lang_arr.iloc[:, 0]=="other", 1, 0)
    
    affixes = ["ma","pa","na","in","an","um","mag","pag","nag","pala","mala","pang","magka", "nagka", "pagka", "pina"]
    df_affixes = pd.DataFrame()
    for affix in affixes:
        df_affixes[affix] = [tweet.count(affix) for tweet in X]
    df_affixes["ma"] = df_affixes["ma"] - df_affixes["mag"] - df_affixes["mala"] 
    df_affixes["pa"] = df_affixes["pa"] - df_affixes["pag"] - df_affixes["pala"] - df_affixes["pang"]  
    df_affixes["na"] = df_affixes["na"] - df_affixes["nag"] 
    df_affixes["mag"] = df_affixes["mag"] - df_affixes["magka"]
    df_affixes["nag"] = df_affixes["nag"] - df_affixes["nagka"]
    df_affixes["pag"] = df_affixes["pag"] - df_affixes["pagka"]    
    
    return df_pos, df_cs, df_lang, df_affixes, df_adjectives
    
def sentiment_tf_method(tweets):
    tf_model = importlib_resources.read_binary("taglish_sentiment", "sentiment_tf.pkl")
    tf_vectorizer = CountVectorizer(stop_words='english', vocabulary=pickle.loads(tf_model))
    tf = tf_vectorizer.fit_transform(tweets)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names
    
def predict_sentiment(X):
    X_processed = sentiment_preprocess(X)
    df_pos, df_cs, df_lang, df_affixes, df_adjectives = sentiment_new_features(X)
    sentiment_tf, sentiment_tf_feature_names = sentiment_tf_method(X_processed)
    sentiment_tf = hstack((coo_matrix(sentiment_tf), coo_matrix(df_pos), coo_matrix(df_lang), coo_matrix(df_cs), coo_matrix(df_affixes), coo_matrix(df_adjectives)))

    sentiment_model = importlib_resources.read_binary("taglish_sentiment", "sentiment_model.sav")
    sentiment_loaded_model = pickle.loads(sentiment_model)
    pred = sentiment_loaded_model.predict(sentiment_tf)

    return pred

def emotion_preprocess(X_arr):
    X_arr_2 = []
    for X in X_arr:
        X = " ".join(re.sub("\n", " ", X).split())
        X = " ".join(re.sub(r"(?:\@|https?\://)\S+", " ", X).split())
        X = " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", X).split())
        X = X.lower()
        X = X.split()
        X = [word for word in X if not word in stop_words]
        with importlib_resources.open_text("taglish_sentiment", "stopwords-tl.json") as file:
            df_stop_words_tl = pd.read_json(file)
        stop_words_tl = df_stop_words_tl.iloc[:,0].tolist()
        X = [word for word in X if not word in stop_words_tl]
        X = [word for word in X if not re.match("(a)*(h)*(ha)+(a)*(h)*", word)]
        X = [word for word in X if not re.match("(e)*(h)*(he)+(e)*(h)*", word)]
        X = [word for word in X if not re.match("(a)(h)(h)*", word)]
        X = [word for word in X if not len(word)==1]
        X = [word for word in X if not word.isdigit()]
        X = " ".join(X)
        X_arr_2.append(X)
    return X_arr_2
    
def emotion_tf_method(tweets):
    tf_model = importlib_resources.read_binary("taglish_emotion", "emotion_tf.pkl")
    tf_vectorizer = CountVectorizer(stop_words='english', vocabulary=pickle.loads(tf_model))
    tf = tf_vectorizer.fit_transform(tweets)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names
    
def emotion_pca_method(X):
  pca_model = importlib_resources.read_binary("taglish_emotion", "emotion_pca.pkl")
  pca = pickle.loads(pca_model)
  emotion_tf_pca = pca.transform(X)
  return emotion_tf_pca
  
def predict_emotion(X):
    X_processed = emotion_preprocess(X)
    emotion_tf, emotion_tf_feature_names = emotion_tf_method(X_processed)
    emotion_tf = normalize(emotion_tf)
    emotion_tf_pca = emotion_pca_method(emotion_tf.todense())

    filepath = resource_filename('taglish_emotion', 'emotion_model.sav')
    emotion_loaded_model = joblib.load(filepath)
    pred = emotion_loaded_model.predict(emotion_tf_pca)

    return pred

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define how the api will respond to the post requests
class SentimentClassifier(Resource):
  def post(self):
    args = parser.parse_args()
    input = args['data']
    predicted = predict_sentiment([input])
    output_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    return output_map[predicted[0]]

class EmotionRegressor(Resource):
  def post(self):
    args = parser.parse_args()
    input = args['data']
    predicted = predict_emotion([input])
    #Love - Joy - Sadness - Fear - Anger - Surprise
    return predicted[0].tolist()

api.add_resource(SentimentClassifier, '/sentiment')
api.add_resource(EmotionRegressor, '/emotion')

if __name__ == '__main__':
  app.run(debug=True, port=8080)
