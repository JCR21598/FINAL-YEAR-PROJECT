
# System related libraries
import copy
import re
import string

# Data Manipulation
import pandas as pd
import numpy as np

# SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# NLTK
from nltk.tokenize import WhitespaceTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# BeautifulSoup


# Visualisation
import matplotlib

# Object Serialization
import pickle

# Other modules of this tool
import ML.ML_functions as ML_funcs


def execute_ML():

    ### Dataset Preparation

    # Select Dataset
    selected_file = "GITHUB"
    available_files = ["github", "kdata", "liar"]

    # Load Datasets as Dataframes
    train_df = ML_funcs.read_dataset(selected_file, available_files, 'train')


    ### Pre-Proprocessing - Cleaning Training Data

    #https://machinelearningmastery.com/clean-text-machine-learning-python/

    # Instantiate object of type Stemmer and gather stop words from nltk
    stemmer = SnowballStemmer("english")
    nltk_stopwords = stopwords.words("english")

    # # Apply stemmer
    # train_df['title-clean'] = train_df['title'].apply(lambda x: [stemmer.stem(y) for y in re.sub("[^a-zA-Z]", " ", str(x)).split()])
    #
    # # Remove stop words
    # train_df['title-clean'] = train_df['title-clean'].apply(lambda x: [item for item in x if item not in nltk_stopwords])

            # # Dealing with NaN values
            # train_df.dropna(inplace=True)  # Delete all of the rows that contain NaN values
            #
            # # Convert all to lower-case
            # train_df['title'] = train_df['title'].str.lower()


    # Split label from train dataframe
    trian_df, train_label = ML_funcs.split_label(selected_file, train_df)

    # Create Validation Set
    X_train, X_test, y_train, y_test = train_test_split(train_df['title'].values.astype('U'), train_label, test_size=0.30, random_state=42)

        # Notes:

        # X -> features/text , y -> labels
        # train -> set used to train , test -> set to test model
        # For instance, y_train translates to = "labels set used to train"

        # For the arguments:
        #       -   test_size:will be 30% of the overall dataset
        #       -   random_state: sets a seed, otherwise,  with each run the data used in train and test will be
        #           different, while at least initially we want it to be constant while model is being tuned


    ### Feature Extraction & Training

    pipeline = Pipeline([("tfidf_vectorizer", TfidfVectorizer(ngram_range=(1,2), stop_words=nltk_stopwords)),
                         ("MNB_classifier", MultinomialNB()) ])
        # Notes:
        #   -   TF-IDF -> weighting factor for features
        #           Perrform both word count and IDF
        #   -

    model = pipeline.fit(X_train, y_train)

    # Grabbing instantes from Pipline to be bale to extract information
    tfidf_vec = model.named_steps["tfidf_vectorizer"]
    MNB_clf = model.named_steps["MNB_classifier"]

    # Test with VALIDATION dataset
    validation_predicted = model.predict(X_test)

    print(f"\n\nAccuracy with {selected_file.lower()} VALIDATION dataset:", accuracy_score(y_test, validation_predicted))

    print(f"\n\nClassification Report for {selected_file.lower()} VALIDATION dataset:")
    print(classification_report(y_test, validation_predicted))

    print(f"\n\nConfusion Matrix for {selected_file.lower()} VALIDATION dataset:")
    print(confusion_matrix(y_test, validation_predicted))

    # Test with TEST dataset
    test_predicted = model.predict

    exit()

    #### Parameter Tuning

    # https://github.com/coding-maniacs/machine_learning_parameter_tuning/blob/master/src/param_tuning.py

    # To tune parameters firstly you need to see what are the available parameteres
    print(pipeline.get_params())

    # Tell the pipeline to try out these parameters, it will try all cominbations and will return what gave the
    # best results
    grid = {
        #example:
        'vectorizer__ngram_range': [(1,1), (2,1)],
        'vectorizer__stop_words':[None, 'english']
    }

    grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='accuracy' , n_jobs=-1, cv=5)
    grid_search.fit(X=train.text, y=y_train)

    # What were the best results
    print("Best Score: " + grid_search.best_score_)
    print("Best Parameteres: " + grid_search.best_params_)



    exit()
    ### TEST

    # Load Test Dataset
    test_df = ML_funcs.read_dataset(selected_file, available_files, 'test')

    # Feature Extraction and Engineering
    test_df['text-tokenized'] = tokenization(train_df, 'text')