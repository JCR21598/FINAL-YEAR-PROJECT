
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

# Visualisation
import matplotlib

# Object Serialization
import pickle


# Other modules of this tool
import ML_functions as ML_funcs
from ML_functions import CustomValidator



class ML_Detector:

    def __init__(self, settings):

        #   setting dictionary is destructed and set to attributes of the object to avoid lengthy expressions/statements
        #   and to use its attributes and methods in different places

        #   General settings
        self.selected_file = settings["general"]["selected_file"]
        self.available_files = settings["general"]["available_files"]

        #   ML settings
        self.validation_set_size = settings["ML_settings"]["validation_set_size"]

        #   Testing settings


    def execute_ML(self):

        ### Data Preparation
        train_df = ML_funcs.read_dataset(self.selected_files, self.available_files, 'train')


        ### Pre-Proprocessing - Cleaning Training Data

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




# Starting point of project
if __name__ == "__main__":

    '''
    #
    #   The mindset with this project is to have made it as versitle and configurable as possible for anyone that wishes
    #   to use the program without understanding the actual programming being used. This format has also considered any
    #   additional developer that wishes to use the program and be able to add new features to the program.
    #
    #   That being said the "setting" dictionary is what allows the user to configure the Machine Learning model to
    #   their liking.
    #
    #   Although it is worth noting that not all key-value pairs are to be edited (unless programmed) and is there to
    #   show what options the user has. But this is still left within the settings as it is convenient for other developers
    #   and just to have all the settings in one place
    #
    '''

    settings = {
        "general":{

            #   select a file from the list of "available_files"
            "selected_file": "GITHUB",
            "available_files" : ["github", "kdata", "liar"],

            #   what happens with NaN values, options: True => consider them, False => Dont consider them
            "considering_NaN": False

        },
        "ML_settings":{

            # Select a number between 0 and 0.9
            "validation_set_size": 0.3

        },
        "testing_settings": {

        },
    }


    '''
    #
    #   For any developer that wishes to extend or modify the code, will most likely need to change this
    #   validation schema variable and customValidator class. Hence it is left here for more convenience.
    #
    #   Cerberus library is used to help the validation process, here is a link to the documentation
    #   URL:    https://docs.python-cerberus.org/en/stable/
    #
    '''

    settings_test = {
        "selected_file": "github"
    }

    schema_test = {
        "selected_file": {"type": "string", "inList": settings['general']['available_files']},
    }

    validator = CustomValidator(schema, allow_unknown=False, require_all=True, empty=False)

    result = validator.validate(settings)
    print(result)


    successful_validation, errors = ML_funcs.setting_validation(settings)

    #   Validation of settings input, if valid continue with run
    if successful_validation:

        #   Intialise and execute Machine Learning approach to detect Fake News
        detector = ML_Detector(settings)
        detector.execute_ML()

    else:


        print("\n\nThe settings provided were not valid, above are the inputs that require modification\n\n")

        exit()
