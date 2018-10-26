import pandas as pd
import numpy as np
import re
import string
import requests, json
from time import time
# dataurl = 'https://data.stadt-zuerich.ch/dataset/zueriwieneu_meldungen/resource/2fee5562-1842-4ccc-a390-c52c9dade90d/download/zueriwieneu_meldungen.json'
# from pandas.io.json import json_normalize
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

stemmer = SnowballStemmer('german')
stopwords = stopwords.words('german')
df = pd.read_csv('data_zueri_wie_neu.csv', sep=';')
df['detail'] = df['detail'].astype(str)
newStopWords = ['dank', 'schon', 'seit', 'fur', 'mehr', 'wurd', 'dank', 'schon', 'br/', 'bitt', 'werd', 'beim', 'dass',
                'ca', 'ja', 'kaum']
stopwords.extend(newStopWords)
X = df.drop('service_code', axis=1)
y = df['service_code']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stopwords)
X_train_vec = vectorizer.fit_transform(X_train['detail'])
X_test_vec = vectorizer.transform(X_test['detail'])


# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train_vec, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    t0 = time()
    pred = clf.predict(X_test_vec)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
    print("classification report:")
    print(metrics.classification_report(y_test, pred))
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))
    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


###########################################################################
results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
        (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50, tol=1e-3),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))