import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import numpy as np


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# response categories type
types = ['aid_products', 'aid_people', 'infras_related', 'weather_related']
aid_products = ['medical_help', 'medical_products', 'water', 'food', 'clothing', 'shelter', 'money']
aid_people = ['search_and_rescue', 'child_alone', 'missing_people', 'refugees', 'security', 'military', 'death']
infras_related = ['infrastructure_related', 'buildings', 'transport', 'electricity', 'tools', 'shops', 'aid_centers', 'hospitals']
weather_related = ['weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather']


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # count for each category
    total_response = df.shape[0]
    n_aid_products = 0
    n_aid_people = 0
    n_infras = 0
    n_weather = 0

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    df_categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories = list(df_categories.keys())
    categories_count = np.zeros(len(categories))
    
    for col in df_categories:
        idx = categories.index(col)
        categories_count[idx] = df_categories[col].sum()
           
        if col in aid_products:
            n_aid_products += df_categories[col].sum()
        elif col in aid_people:
            n_aid_people += df_categories[col].sum()
        elif col in infras_related:
            n_infras += df_categories[col].sum()
        elif col in weather_related:
            n_weather += df_categories[col].sum()
        else:
            continue
    
    # top 10 message most received
    categories_sorted_idx = np.argsort(categories_count)[-10:]
    top_10_count = []
    top_10_cat = []
    for idx in categories_sorted_idx:
        top_10_count.append(categories_count[idx])
        top_10_cat.append(categories[idx])
        
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=types,
                    y=[n_aid_products, n_aid_people, n_infras, n_weather]
                )
            ],

            'layout': {
                'title': 'Distribution of Message Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Types"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_10_cat,
                    y=top_10_count
                )
            ],

            'layout': {
                'title': 'Top 10 most received Message Category',
                'yaxis': {
                    'title': "Count"
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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()