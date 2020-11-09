import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
import pickle

#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', "stopwords"])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', con=engine)
    #df = df[["message", "medical_help", "offer", "search_and_rescue"]]
    #df = df[(df["medical_help"] != 0) | (df["offer"] != 0) | (df["search_and_rescue"] != 0)]

    #X = df.iloc[:, 0]
    #Y = df.iloc[:, 1:]

    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    print(X.shape, Y.shape)

    return [X, Y, category_names]


def tokenize(text):
    text_ = re.sub(r'[\?\.,!"\'\*\(\)\-\&\$\\\/+\:\;\<\>\=#%0-9]?', " ", text.lower())

    stopwords_ = stopwords.words('english')
    tokens = [word_tokenize(i) for i in text_]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for sentence in tokens:
        for tok in sentence:
            if not tok in stopwords_:
                clean_tok = lemmatizer.lemmatize(tok).lower().strip()
                clean_tokens.append(clean_tok)

    clean_tokens = pd.Series(clean_tokens).drop_duplicates().tolist()

    return clean_tokens


def build_model():
    pipeline = Pipeline(
        [("vect", CountVectorizer(tokenizer=tokenize)),
         ('tfidf', TfidfTransformer()),
         ("clf", MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
         ])

    parameters = {
        'clf__estimator__min_samples_split': [2, 4, 6],
        'clf__estimator__min_samples_split': [100, 500, 1000],
        'clf__estimator__n_estimators': [10, 50, 100]


    }

    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    #global result
    #result = classification_report(Y_test, Y_pred)
    try:
        pd.DataFrame(classification_report(Y_test, Y_pred, output_dict=True)).to_csv("classification_score.csv")
        pd.DataFrame(model.cv_results_).to_csv("CV_results.csv")
    except:
        pd.DataFrame(model.cv_results_).to_csv("CV_results.csv")


    return


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

    return


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()