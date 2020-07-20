#
#       Author:     Juan Camilo Rodriguez
#
#   About File:     This file is for the user to choose whether to perform training or testing. Upon that can set the
#                   settings for either. Will have a range of options to choose from and able to make various combinations.
#
#                   The program has the ability to allow new datasets to be added with little to no chnage to the programming files.
#                   For instance the user can add a new dataset and use it, as long as it follows the set standard. In addition,
#                   can train a model from a range of classifiers and then save it if wished. After is able to load the model
#                   again and challenge the classifier with numerous testing options.


'''
#
#   The mindset with this project is to have made it as versatile and configurable as possible for anyone that wishes
#   to use the program without understanding the actual programming being used. This format has also considered any
#   additional developer that wishes to use the program and be able to add new features to the program.
#
#   That being said the "config" file is what allows the user to configure the Machine Learning model and Dataset(s) to
#   their liking.
#
#   Here are some notes about the following dictionaries:
#
#       program_operation:  This dictionary is to set main the main operations of the program, which is to either do training
#                           or to test some models
#
#       dataset_settings:   This dictionary features the chracteristics of each dataset so that the program is able to understand
#                           how each dataset works
#
#       training_settings:  This dictionary focuses on setting the values that the machine learning will use to train the
#                           model.
#
#       testing_settings:   This dictionary is to establish the steps that the program should take for the testing of a model
#
'''


program_operations ={

    "operation": "train",
    "available_operations": ["train", "test"],
    "export_model": True,

}


dataset_settings = {
    "github": {
        "input": {
            "title": True,
            "body-text": True,
        },
        "label": {
            "real_news_label": [0],
            "fake_news_label": [1],
            "label-column-index": 1
        },
    },
    "kdata": {

    },
    "liar":{

    },

}




training_settings = {

    "models": ["model"],

    "dataset": {

        #   select a file from the list in dataset_settings
        "selected_file": "github",
        "available_files": dataset_settings.keys()

    },

    "ML_settings": {

        #  Either use "stemmer" or "lemmatizer" or False for not applying anything
        "apply_stemmer_or_lemmatizer": False,

        #   Stopwords - words that are common and have no significatn value
        "apply_stopword_remover": False,

        #   what happens with NaN values, options: True => consider them, False => Dont consider them
        "considering_NaN": False,

        # Select a number between 0 and 0.9
        "validation_set_size": 0.3,

    },
}


testing_settings = {

    "model": "MOVETE",

}