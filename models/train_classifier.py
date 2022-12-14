import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    Load data from sqlite3 db using pandas API
    
    Params:
        - database_filepath (str): .db file path which have the data you want to load
    Return:
        A Pandas DataFrame
    '''
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table('DisasterResponse', engine)
        #print(df.columns)
        X = df['message'] 
        Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

        # return X, Y and categories names
        #print(X.shape)
        #print(Y.shape)
        return X, Y, Y.keys()
    except:
        print('ERROR: Got some error while loading data')
        return None, None, None

    
def tokenize(text):
    '''
    Tokenize text sentence into lemmatized words
    
    Params:
        - text (str): text sentence
    Return:
        List of tokenized lemmatized words
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build model with Pipeline, using CountVetorizer with tokenize function, 
    TfidfTransformer and use MultiOutputClass with RandomForest 
    for classify multiple output class
    
    Return:
        a Pipeline object
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('mcl', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model by using metrics f1 score, precision and recall 
    for each output category of the dataset
    
    Params:
        - X_test: data for testing
        - Y_test: data for testing
        - category_names: list of category in dataset
    '''
    Y_pred = model.predict(X_test)
    
    for idx, column in enumerate(category_names):
        print(f'INFO: {column}: {classification_report(Y_test.values[:,idx], Y_pred[:,idx])}')


def save_model(model, model_filepath):
    '''
    Save pretrained model into pickle file with given file path
    
    Params:
        - model: pretrained model
        - model_filepath: desired pickle file path 
    '''
    try:
        pickle.dump(model, open(model_filepath,'wb'))
        print('DEBUG: Saved model successfully')
    except:
        print('ERROR: Got error while saving model to pickle')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('INFO: Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('INFO: Building model...')
        model = build_model()
        
        print('INFO: Training model...')
        model.fit(X_train, Y_train)
        
        print('INFO: Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('INFO: Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('INFO: Trained model saved!')

    else:
        print('ERROR: Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '        \
              'save the model to as the second argument. \n\nExample: python '       \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()