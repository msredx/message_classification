# Disaster Response Pipeline Project

### Overview:
Text Messages from several real disasster scenarios are used to train classification of messages into specific categories relevant for disaster response. The use case would be to coordinate agencies / organization for quick reponses and optimize dissemination of help during/ after a disaster.
This project showcases 
- an ETL pipeline (loading data, cleaning data, storing data to an sql-databse)
- a machine learning pipeline (loading data from SQL database, feature extraction, training a classifier, grid search, evaluating model performance and saving the model)
- a flask-based web-app to apply the classifier to a single message and give some data visualizations

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves a pkl-file containing the trained model to directory "models"
        `python models/train_classifier.py data/DisasterResponse.db models`


2. Run the following command in the project's root directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

### Requirements:
- Python 3.5+ (tested with 3.6.3 and 3.7.3)
- numpy, scipy, pandas, scikit_learn, nltk, flask, plotly, SQLAlchemy

### File structure
`
- data
|- disaster_categories.csv  # file containing category labels for messages
|- disaster_messages.csv  #  file containing messages
|- process_data.py  # script to preprocess data (read, clean, export to SQL-database)
|- DisasterResponse.db   # saved database where cleaned data is stored

- models
|- train_classifier.py  # script to train the classifier
|- classifier.pkl  # saved model
|- classifier_metrics.pkl  # saved metrics for model

- app
| - template   #template folder for flask web-app
| |- master.html  # main page 
| |- go.html  # result page
|- run.py  # script to run flask server

- README.md
- LICENSE  
- requirements.txt  # list of requirements (generated with [pipreqs](https://pypi.org/project/pipreqs/))

*** Acknowledgments
The dataset is from [Figure eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/), as featured in the [Udacity](http://www.udacity.com) course on [data science](https://www.udacity.com/course/data-scientist-nanodegree--nd025), which also provided some code snippets for this project.

*** Comments
The dataset is pretty unbalanced for most categories, with some categories being "rare events" (<0.5 %) , and performance of the chosen SGDclassifier is especially bad for these rare categories (ROC-AUC ~ 0.5), but quite ok for others (up to 0.9). During training this was already accounted for by optimizing hyperparameters for high ROC-AUC values, however with limited success (mean roc-auc across categories ~.65 to .70)
In a real life situation, the classifier should be optimized for detection of rare events.
Probably, pretrained deep learning networks could be fine tuned for optimal performance on this dataset (e.g. [BERT](https://github.com/google-research/bert) or [MT-DNN](https://github.com/namisan/mt-dnn)) 

