import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine
import nltk
#nltk.download(['punkt', 'wordnet','stopwords'])


app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    extract information whether text starts with verb or verbal phrase
    can be used as estimator in sklearn (transform)
    returns:
    0 or 1
    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
            except:
                return 0
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    '''
    simple tokenization: keep only chars and numbers, convert to lowercase, tokenize and lemmatize using nltk
    text: str that will be tokenized

    returns new_tokens (list of extracted tokens)
    '''

    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #get tokens
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    new_tokens = []
    for tok in tokens:
        new_tokens.append(lemmatizer.lemmatize(tok).strip())
    return new_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("models/classifier.pkl")
metrics = joblib.load("models/classifier_metrics.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/',methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)
    category_names = list(df.columns[4:])
    #category_roc_auc = list(metrics['roc_auc'])

    def get_category_values(cat_str):
        '''
        cat_str: list of category names
        returns x_val and y_val for plot
        '''
        cat_df = df[df.columns[df.columns.isin(cat_str)]].sum()/df.shape[0]
        y_perc= list(cat_df)
        y_auc = list(metrics[metrics.index.isin(cat_str)]['roc_auc'])
        return  y_perc, y_auc

    # Parse the POST request categories list
    if (request.method == 'POST') and request.form:
        #figures = return_figures(request.form)
        categories_selected = []
        for cat in request.form.lists():
            categories_selected.append(cat[1][0])
        y_perc, y_auc = get_category_values(categories_selected)
        print(categories_selected)
        print(y_perc)
        print(y_auc)


    # GET request returns all categories for initial page load
    else:
    #figures = return_figures()
        categories_selected = []
        for cat in category_names:
            categories_selected.append(cat)
        print(categories_selected)
        y_perc, y_auc = get_category_values(categories_selected)
        print(y_perc)
        print(y_auc)
        #print(all_categories)


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [

        {
            'data': [
                Bar(
                    x=categories_selected,
                    y=y_perc
                )
            ],

            'layout': {
                'title': 'Relative number of messages per category',
                'yaxis': {
                    'title': "perc (count_category/nr_messages)"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=categories_selected,
                    y=y_auc
                )
            ],

            'layout': {
                'title': 'classifier quality',
                'yaxis': {
                    'title': "area under the curve (roc): 0.5 is chance, 1 is perfect",
                    'showgrid': True,
                    'zeroline': False,
                    'range': [0.5, 1]
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },

        {
            'data': [
                Scatter(
                    x=y_perc,
                    y=y_auc,
                    mode="markers"
                )
            ],

            'layout': {
                'title': 'rel number versus classifier quality',
                'yaxis': {
                    'title': "auc"
                },
                'xaxis': {
                    'title': "perc (count_category/nr_messages)"
                }
            }
        },




    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids,
        graphJSON=graphJSON,
        all_categories=category_names,
        categories_selected=categories_selected)


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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
