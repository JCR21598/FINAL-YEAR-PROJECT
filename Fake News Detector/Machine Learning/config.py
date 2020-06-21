'''
#
#   The mindset with this project is to have made it as versatile and configurable as possible for anyone that wishes
#   to use the program without understanding the actual programming being used. This format has also considered any
#   additional developer that wishes to use the program and be able to add new features to the program.
#
#   That being said the "config" file is what allows the user to configure the Machine Learning model and Dataset(s) to
#   their liking.
#
#   Here are some notes about the follwing dictionaries:
#
#       ML_settings:        This dictionary focuses on setting the values that the machine learning will use to train the
#                           model.
#
#       dataset_settings:   This dictionary is less to be manipulated by the user but more about establishing information
#                           about a dataset which then the model uses. Additionally, this also allows a place for opther
#                           developers to add more datasets as long as it follows the required information that the model needs
#
'''

program_operations ={

    "operation": "test",
    "available_operations": ["train", "test"],
    "export_model": True,

}


ML_settings = {

    "general": {

        #   select a file from the list of "available_files"
        "selected_file": "liar",
        "available_files": ["github", "kdata", "liar"],

    },

    "ML_settings": {

        #  Stemmer - reduction of variations of words
        "apply_stemmer": False,

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


dataset_settings = {
    "github": {
        "label_types": [0, 1]
    },
    "kdata": {

    },
    "liar":{

    },

}