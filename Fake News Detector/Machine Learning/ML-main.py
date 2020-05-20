
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
#from ML_functions import CustomValidator



class ML_Detector:

    def __init__(self, settings):

        #   setting dictionary is destructed and set to attributes of the object to avoid lengthy expressions/statements
        #   and to use its attributes and methods in different places

        #   General settings
        self.selected_file = settings["general"]["selected_file"]
        self.available_files = settings["general"]["available_files"]

        #   ML settings
        self.apply_stopword_remover = settings["ML_settings"]["apply_stopword_remover"]
        self.apply_stemmer = settings["ML_settings"]["apply_stemmer"]
        self.considering_NaN = settings["ML_settings"]["considering_NaN"]
        self.validation_set_size = settings["ML_settings"]["validation_set_size"]

        #   Testing settings


    def execute_ML(self):

        ###     Data Preparation
        train_df = ML_funcs.read_dataset(self.selected_file, self.available_files, 'train')


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


        # Split label from train dataframe
        trian_df, train_label = ML_funcs.split_label(self.selected_file, train_df)

        # Create Training and Validation Set
        X_train, X_test, y_train, y_test = train_test_split(train_df['title'].values.astype('U'), train_label, test_size=self.validation_set_size, random_state=42)


        print(y_train)
            # Information:
            #   X -> features/text , y -> labels
            #   train -> set used to train , test -> set to test model
            # Therefore, y_train translates to = "labels set used to train"

            # For the arguments:
            #       -   test_size: will be 30% of the overall dataset
            #       -   random_state: sets a seed, otherwise,  with each run the data used in train and test will be
            #           different, while at least initially we want it to be constant while model is being tuned


        # with pd.option_context('display.max_rows', None, 'display.max_columns',None):  # more options can be specified also
        #     print(train_df)



        ###     Feature Extraction/Engineering & Pipelining

        tfidf_vectorizer = TfidfVectorizer()
        multinomialNB = MultinomialNB()

        tfidf_vectorizer_params = tfidf_vectorizer.get_params()
        multinomialNB_params = multinomialNB.get_params()

        ML_funcs.print_params("TFIDF", tfidf_vectorizer_params)
        ML_funcs.print_params("MultinomialNB", multinomialNB_params)

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

        # Create the model from training data
        model = grid_search.fit(X=X_train, y=y_train)


        # Best Hyper-parameters
        print("Best Hyper-parameters:")
        print(grid_search.best_params_)



        # Test with VALIDATION dataset
        validation_predicted = model.predict(X_test)

        print(f"\n\nAccuracy with {self.selected_file.lower()} VALIDATION dataset:", accuracy_score(y_test, validation_predicted))

        print(f"\n\nClassification Report for {self.selected_file.lower()} VALIDATION dataset:")
        print(classification_report(y_test, validation_predicted))

        print(f"\n\nConfusion Matrix for {self.selected_file.lower()} VALIDATION dataset:")
        print(confusion_matrix(y_test, validation_predicted))



        # Real Data Test

        test = np.array([
                         "Is It Dangerous to Wear a COVID-19 Protective Mask for Too Long?",
                         "Are Bill Gates and the ID2020 Coalition Using COVID-19 To Build Global Surveillance State?",
                         "Are Evolution Fresh Drinks ‘Poison’?",
                         "NBA Star Kobe Bryant Killed in Helicopter Crash",
                         "Is a Testicular Blow Exponentially More Painful Than Childbirth?",
                         "Does the McCarran-Walter Act of 1952 Bar Muslims from Holding Public Office?",
                         "Was Emily Jones Decapitated by a Somali Migrant?",
                         "Did Harvard Admit Michelle Obama's' Degrees Are Fake?",
                         "Did Self-Described Psychic Sylvia Browne Predict the Coronavirus?",
                         "Did Trump Administration Fire the US Pandemic Response Team?"
                         "Did Trump Tweet in 2009 That He Would ‘Never Let Thousands of Americans Die From a Pandemic’?"
                        ])

        test_label = np.array([0,0,0,1,0,0,0,1,1,0])

        test_df = pd.Series(test)
        test_label_df = pd.Series(test_label)

        real_data_predicted = model.predict(test_df)

        print(f"\n\nAccuracy with REAL dataset:",
              accuracy_score(test_label_df, real_data_predicted))

        print(f"\n\nClassification Report for REAL dataset:")
        print(classification_report(test_label_df, real_data_predicted))

        print(f"\n\nConfusion Matrix for REAL dataset:")
        print(confusion_matrix(test_label_df, real_data_predicted))



        # Data From other dataset

        test2 = np.array([
            "Building a wall on the U.S.-Mexico border will take literally years.",
            "Wisconsin is on pace to double the number of layoffs this year.",
            "The Fed created $1.2 trillion out of nothing, gave it to banks, and some of them foreign banks, so that they could stabilize their operations.",
            "Texas families have kept more than $10 billion in their family budgets since we successfully fought to restore Texas sales tax deduction a decade ago.",
            "A salesclerk at Hobby Lobby who needs contraception is not going to get that service through her employers health care plan because her employer doesnt think she should be using contraception.",
            "When undocumented children are picked up at the border and told to appear later in court ... 90 percent do not then show up.",
            "A strong bipartisan majority in the House of Representatives voted to defund Obamacare.",
            "Says Milken Institute rated San Antonio as nations top-performing local economy.",
            "When Atlanta Police Chief George Turner was interim head of the department, overall crime fell 14 percent and violent crime dropped 22.7 percent.",
            "If you want to vote in Texas, you can use a concealed-weapon permit as a valid form of identification, but a valid student ID isnt good enough.",
            "Barack Obama 'rejects everyone white, including his mother and his grandparents.'",
            "In the early 1980s, Sen. Edward Kennedy secretly offered to help Soviet leaders counter the Reagan administrations position on nuclear disarmament.",
            "There is no record of congresswoman Betty Sutton ... ever holding a single in-person town hall meeting open to the general public.",
            "Says state Rep. Sandy Pasch, her recall opponent, voted to allow public school employees to use taxpayer dollars to pick up the tab forViagra.",
            "If we ban the practice of earmarks, we could save the American taxpayer anywhere between $15 (billion) to $20 billion dollars a year in pork-barrel spending.",

        ])

        test_label2 = np.array([1,0,1,1,1,0,0,1,1,1,0,1,0,0,1])

        test_df2 = pd.Series(test)
        test_label_df2 = pd.Series(test_label)

        real_data_predicted2 = model.predict(test_df2)

        print(f"\n\nAccuracy with OTHER dataset:",
              accuracy_score(test_label_df2, real_data_predicted2))

        print(f"\n\nClassification Report for OTHER dataset:")
        print(classification_report(test_label_df2, real_data_predicted2))

        print(f"\n\nConfusion Matrix for OTHER dataset:")
        print(confusion_matrix(test_label_df2, real_data_predicted2))


        # # Test with TEST dataset
        # test_predicted = model.predict
        #
        # ### TEST
        #
        # # Load Test Dataset
        # test_df = ML_funcs.read_dataset(selected_file, available_files, 'test')
        #
        # # Feature Extraction and Engineering
        # test_df['text-tokenized'] = tokenization(train_df, 'text')


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

        },
        "ML_settings":{

            #  Stemmer - reduction of variations of words
            "apply_stemmer" : False,

            #   Stopwords - words that are common and have no significatn value
            "apply_stopword_remover": False,

            #   what happens with NaN values, options: True => consider them, False => Dont consider them
            "considering_NaN": False,


            # Select a number between 0 and 0.9
            "validation_set_size": 0.3,

        },
        "testing_settings": {

        },
    }

    detector = ML_Detector(settings)
    detector.execute_ML()















    '''
    #
    #   For any developer that wishes to extend or modify the code, will most likely need to change this
    #   validation schema variable and customValidator class. Hence it is left here for more convenience.
    #
    #   Cerberus library is used to help the validation process, here is a link to the documentation
    #   URL:    https://docs.python-cerberus.org/en/stable/
    #
    '''

    # settings_test = {
    #     "selected_file": "github"
    # }
    #
    # schema_test = {
    #     "selected_file": {"type": "string", "inList": settings['general']['available_files']},
    # }
    #
    # validator = CustomValidator(schema, allow_unknown=False, require_all=True, empty=False)
    #
    # result = validator.validate(settings)
    # print(result)

    # successful_validation, errors = True            #ML_funcs.setting_validation(settings)
    #
    # #   Validation of settings input, if valid continue with run
    # if successful_validation:
    #
    #     #   Intialise and execute Machine Learning approach to detect Fake News
    #
    #
    #
    # else:
    #
    #
    #     print("\n\nThe settings provided were not valid, above are the inputs that require modification\n\n")
    #
    #     exit()







