
# System related libraries
import copy
import re
import string
import time

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

# NLTK
from nltk.tokenize import WhitespaceTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Object Serialization
import joblib

# Other modules of this tool
import functions as funcs
#from ML_functions import CustomValidator



class Detector:

    def __init__(self, settings):

        #   setting dictionary is destructed and set to attributes of the object to avoid lengthy expressions/statements
        #   and to use its attributes and methods in different places

        #   Program Operation
        self.operation = settings["program_operation"]["operation"]
        self.available_files = settings["program_operation"]["available_operations"]

        #   General settings
        self.selected_file = settings["general"]["selected_file"]
        self.available_files = settings["general"]["available_files"]

        #   ML settings
        self.apply_stopword_remover = settings["ML_settings"]["apply_stopword_remover"]
        self.apply_stemmer = settings["ML_settings"]["apply_stemmer"]
        self.considering_NaN = settings["ML_settings"]["considering_NaN"]
        self.validation_set_size = settings["ML_settings"]["validation_set_size"]


    @funcs.train_time
    def train_model(self):

        ###     Data Preparation
        train_df = funcs.read_dataset(self.selected_file, self.available_files, self.operation)


        ###     Pre-Proprocessing - Cleaning Training Data

        # Stemmer
        if self.apply_stemmer:
            stemmer = SnowballStemmer("english")
            train_df['title-clean'] = train_df['title'].apply(lambda x: [stemmer.stem(y) for y in re.sub("[^a-zA-Z]", " ", str(x)).split()])

        # Stopwords
        if self.apply_stopword_remover:
            nltk_stopwords = stopwords.words("english")
            train_df['title-clean'] = train_df['title-clean'].apply(lambda x: [item for item in x if item not in nltk_stopwords])

        # NaN Values
        if self.considering_NaN:
            train_df.dropna(inplace=True)  # Delete all of the rows that contain NaN values


        ###     Validation Set

        # Split label from train dataframe
        trian_df, train_label = funcs.split_label(self.selected_file, train_df)

        # Create Training and Validation Set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_df['title'].values.astype('U'), train_label, test_size=self.validation_set_size, random_state=42)

            # Information:
            #   X -> features/text , y -> labels
            #   train -> set used to train , test -> set to test model
            # Therefore, y_train translates to = "labels set used to train"

            # For the arguments:
            #       -   test_size: will be 30% of the overall dataset
            #       -   random_state: sets a seed, otherwise,  with each run the data used in train and test will be
            #           different, while at least initially we want it to be constant while model is being tuned



        ###     Feature Extraction/Engineering & Pipelining

        tfidf_vectorizer = TfidfVectorizer()
        multinomialNB = MultinomialNB()

        tfidf_vectorizer_params = tfidf_vectorizer.get_params()
        multinomialNB_params = multinomialNB.get_params()

        funcs.print_params("TFIDF", tfidf_vectorizer_params)
        funcs.print_params("MultinomialNB", multinomialNB_params)

        pipeline = Pipeline([("tfidf_vectorizer", tfidf_vectorizer),
                             ("MNB_classifier", multinomialNB) ])

            # Notes:
            #   -   TF-IDF -> weighting factor for features
            #           Perform both word count and IDF
            #   -

        ####    Hyper-Parameter Tuning & Training

        # https://github.com/coding-maniacs/machine_learning_parameter_tuning/blob/master/src/param_tuning.py

        # Tell the pipeline to try out these parameters, it will try all combinations and will return what gave the
        # best results
        grid = {
            #   parameters for tf-idf_vectorizer
            'tfidf_vectorizer__ngram_range': [(1, 1), (1, 2), (2,2), (1,3), (3,3), (1,10)],
            'tfidf_vectorizer__stop_words': [None, 'english'],
            'tfidf_vectorizer__binary' : [True, False],
            'tfidf_vectorizer__max_features' : [5, 15, 50, None],
            "tfidf_vectorizer__smooth_idf": [True, False],
            "tfidf_vectorizer__sublinear_tf": [False, True],

            #   parameters for MNB_classifier
            "MNB_classifier__alpha":[1.0],
            "MNB_classifier__fit_prior": [True],

        }

        # GridSearchCV just works as a another model
        grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='accuracy', n_jobs=-1, cv=5)

        # Best Hyper-parameters that Grid Search found
        # print("Best Hyper-parameters:")
        # print(grid_search.best_params_)

        # Create the model from training data
        self.model = grid_search.fit(X=self.X_train, y=self.y_train)

        return self.model



    ###     Visualisation Methods

    # Print the entire Dataframe
    def print_all_df(self, dataframe):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(dataframe)


    def print_selected_dataframe(self, dataframe):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(dataframe)

