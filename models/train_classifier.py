import sys
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import re
import nltk
from nltk.corpus import stopwords

def load_data(database_filepath):
    '''
    Loading and preproccessing
    Input: database_filepath
    Output: X,y, and teh category_names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages_tbl', engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'],axis=1)
    category_names = y.columns
    return X, y, category_names
 
    


def tokenize(text):
    '''
    tokenize and lammenize the text
    Input: text
    output: ready to go list  of lamminized words which are not stop words
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]

    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemm


def build_model():
    '''
    Developing the pipeline with agrid search
    
    Output: model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
        #'clf__estimator__min_samples_split': [2, 4]
        } 
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model
    Inputs: model, X_test, Y_test, category_names
    Outputs: accuracy and the evaluation report
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))
    


def save_model(model, model_filepath):
    '''
    saving the model
    Input: model, model_filepath
    Output: saving the model into a pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()