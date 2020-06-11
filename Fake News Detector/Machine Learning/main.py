
from detector import Detector
from testing import Testing

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
        
        "program_operation" :{

            "operation": "train",
            "available_operations": ["train", "test"],
        },

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

    '''
        #
        #   Program was developed in Object-Oriented format. This relieved a lot of early developing stages which was a 
        #   program that was hard to track, unsustainable and hard to unmaintainable.
        #
    '''


    #   This is you have a new model that needs training - worth noting that when training it will export the model
    if settings["program_operation"]["operation"] is "train":

        #   Instatiating objects to use
        detector = Detector(settings)

        #   Train model
        detector.train_model()



    #   This is if you already a model and just want to test it
    elif settings["program_operation"]["operation"] is "test":

        tester = Testing()

        #   Test the model
        tester.test_validation()
        tester.test_recent()
        tester.test_other_datasets()


    else:
        print(f"The program operation {program_operation} is not valid")










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







