import json
import plotly
import pandas as pd
import numpy as np


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)



def tokenize(text):
    """
    Takes input takes and returns clean tokens for classification.

    :param text: input text
    :return: cleaned tokens of input
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

#load evaluation_scores

scores = pd.read_csv("/Users/nassimgharbi/Documents/Udacity-Data-Science_Nano/Week-3_Data_Engineering/disaster_response_pipeline_project/classification_score.csv").transpose()
scorers = scores.loc["Unnamed: 0",:].to_list()
scores_vals = np.array(scores.drop("Unnamed: 0", axis=0)).reshape(-1,4)
scores_vals_labels = np.array(scores[-4:].reset_index())


cols = df.columns[4:]



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    

    category = df[cols].columns.tolist()
    counts = df[cols].sum()



    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category,
                    y=counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    table = request.args.get('table', "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    classification_result = dict(zip(cols, classification_labels))
    scoring_results = dict(zip(cols, scores_vals))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_result,
        scorers = scorers,
        scoring_results = scoring_results,
        scores_vals_labels = scores_vals_labels,
        table = table
    )



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()