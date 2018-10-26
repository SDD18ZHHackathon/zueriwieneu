from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def train_model(documents, classes):

    stopword = stopwords.words('german')
    newStopWords = ['danke', 'schon', 'seit', 'fur', 'für', 'mehr', 'wurde', 'dank', 'schon', 'br/', 'bitte', 'werde', 'beim',
                    'dass',
                    'ca', 'ja', 'kaum']
    stopword.extend(newStopWords)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stopword)

    documents_vec = vectorizer.fit_transform(documents)

    #clf = RidgeClassifier(tol=1e-2, solver="sag")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(documents_vec, classes)

    return((vectorizer, clf))


def predict(text, models):
    vectorizer, clf = models
    text_vec = vectorizer.transform(text)
    pred = clf.predict_proba(text_vec) # use just 'prediction' for the RidgeClassifier
    return pred

