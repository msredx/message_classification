import sys
import os
# import libraries
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scipy.stats.mstats import gmean
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, fbeta_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
## define some custom stopwords
#full stopwords from nltk
stopwords_a= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
              "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
              'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
              'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
              'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
              'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
              'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
              'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
              'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
              'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
              'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
              'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
              'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
              'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
              "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
              'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

#customized stopwords from nltk, verbs leftout
stopwords_b= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
              "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
              'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
              'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
              'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
              'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
              'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
              'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
              'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
              'just','now', 'd', 'll', 'm',
              'o', ]

#customized stopwords from nltk, questwords and "in" , "between", etc. left out
stopwords_c= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
              "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
              'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
              'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
              'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
              'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
              'as', 'until', 'while', 'of', 'at', 'by', 'for', 'then', 'once',  'there',
              'all', 'any', 'both', 'each', 'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
              'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
              'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
              'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
              'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
              'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
              "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
              'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


#customized stopwords only pronouns & articles, sentence combiner
stopwords_d= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
       "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
       'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
       'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
       'their', 'theirs', 'themselves',
       'this', 'that', "that'll", 'these', 'those','a', 'an', 'the', 'and',
       'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
       'by', 'for', 'with', 'about', 'against']


def load_data(database_filepath):
    '''
    loads data from sql-database
    database_filepath: path to sqlite database
    returns X (message text), Y(multiple binarized categories), list of category names
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql('SELECT * FROM messages', con = engine)
    X = df['message']
    Y = df.drop(['genre', 'id', 'original', 'message'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names

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


def build_model():
    '''
    define pipeline and/or gridsearch object for feature extraction and trainig classifier
    returns pipeline or gridsearch object
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor()),
        ])),
        ('clf', MultiOutputClassifier(SGDClassifier()))
    ])

#parameters = {'features__tfidf_pipeline__vect__max_df': (0.6, 0.8, 1),
#              'features__tfidf_pipeline__vect__ngram_range': ((1,1),(1, 2)),
#              'features__tfidf_pipeline__vect__stop_words': (stopwords_a,stopwords_b),
#              'features__tfidf_pipeline__vect__max_features': (None, 10000),
#              'clf__estimator__max_iter': (50,),
#              'clf__estimator__alpha': (0.00001,),
#              'clf__estimator__penalty': ('elasticnet','l2')}

    parameters = {'features__tfidf_pipeline__vect__max_df': (0.6,),
              'features__tfidf_pipeline__vect__ngram_range': ((1, 2),),
              'features__tfidf_pipeline__vect__stop_words': (stopwords_a,),
              'features__tfidf_pipeline__vect__max_features': (None,),
              'clf__estimator__max_iter': (50,),
              'clf__estimator__alpha': (0.00001,),
              'clf__estimator__penalty': ('elasticnet',)}
    cv = GridSearchCV(pipeline, param_grid = parameters, cv=5, n_jobs=1,
                      verbose = 2, scoring = make_scorer(roc_auc_score))

    return cv
    #return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model
    prints evaluation metrics
    '''
    def get_metrics (y_test, y_pred):
        '''
        runs a number of metrics on multioutput classifier results
        y_test: dataframe with true labels (binary)
        y_pred: numpy array with predicted labels (y_pred = XXXX.predict(X_test) from an sklearn estimator)

        returns: dataframe with accuracy, precision, f1, recall, tp, tn, fp, fn, roc_auc



        scores for each multioutput classifier
        '''
        accuracy, precision, recall, f1, support, tn, fp, fn, tp, roc_auc = [], [], [], [], [], [], [], [], [], []
        for i in range (len(y_pred[0,:])):
            try:
                accuracy.append(accuracy_score(y_test.iloc[:,i],y_pred[:,i]))
            except:
                accuracy.append(np.nan)
            try:
                precision.append(precision_score(y_test.iloc[:,i],y_pred[:,i]))
            except:
                precision.append(np.nan)
            f1.append(f1_score(y_test.iloc[:,i],y_pred[:,i]))
            recall.append(recall_score(y_test.iloc[:,i],y_pred[:,i]))
            confusion_mat = confusion_matrix(y_test.iloc[:,i],y_pred[:,i])
            #see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
            tn_, fp_, fn_, tp_ = confusion_mat.ravel()
            tn.append(tn_)
            fp.append(fp_)
            fn.append(fn_)
            tp.append(tp_)
            roc_auc.append(roc_auc_score(y_test.iloc[:,i],y_pred[:,i]))
        metrics = pd.DataFrame({'cat':category_names,'accuracy':accuracy, 'precision':precision,
                                'f1':f1, 'recall':recall,'true_pos': tp, 'true_neg': tn, 'false_pos':fp,
                                'false_neg':fn, 'roc_auc':roc_auc})
        metrics.set_index(keys='cat', inplace=True)
        return metrics
        #print(f"Accuracy: {accuracy}")
        #print(f"Precision: {precision}")
        #print(f"Recall: {recall}")
        #print(f"fscore: {fscore}")
        #print(f"support: {support}")

    Y_pred_test=model.predict(X_test)
    test_metrics=get_metrics(Y_test,Y_pred_test)
    #we take the mean of all metrics, because we want all predictors to be good,
    #irrespective of their relative occurance. This is equivalent to macro-averaging of scores
    # for the binary multilabel case
    print("metrics for test set:")
    print(test_metrics.mean())
    print("metrics for test set, each category")
    print(test_metrics)
    return test_metrics

def save_model(model, metrics, model_filepath, metrics_filepath):
    '''
    save model and metrics to pkl file
    '''
    joblib.dump(model, model_filepath)
    joblib.dump(metrics, metrics_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_path = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        metrics = evaluate_model(model, X_test, Y_test, category_names)

        metrics_filepath = os.path.join(model_path,'classifier_metrics.pkl')
        model_filepath = os.path.join(model_path,'classifier.pkl')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, metrics, model_filepath, metrics_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
