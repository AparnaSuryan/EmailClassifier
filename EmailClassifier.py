
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

df = pd.read_csv("spam.csv", encoding = 'latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], 
        axis = 1, inplace = True)
df.rename(columns = {'v1': 'labels', 'v2': 'email'}, inplace = True)
mappings = {'ham': 0, 'spam': 1}
df['new_label'] = df['labels'].map(mappings)
df.drop(['labels'], axis = 1, inplace = True)

def model(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    return classifier, accuracy, X_test, y_test

pipeline = Pipeline([('vectorizer', TfidfVectorizer()), 
                     ('classifier', SVC()),
                     ])
 
classifier, accuracy, X_test, y_test = model(pipeline, df['email'], 
                                             df['new_label'])
joblib.dump(classifier, 'model.pkl')

#Load trained model

trained_model = joblib.load('model.pkl')
label = trained_model.predict([X_test[4805]])
prediction = dict((v,k) for k, v in mappings.items())[label[0]]