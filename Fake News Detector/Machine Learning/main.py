#
#       Author:     Juan Camilo Rodriguez
#
#   About File:     Execution file. Uses heavily the config file to execute other files.
#

"""===     IMPORTS     ==="""

'''Third-party Imports'''


'''In-built Imports'''
import logging
import joblib

'''Personal Imports'''
from training import Training
from testing import Testing
from config import program_operations, training_settings, testing_settings, dataset_settings
import functions as funcs



if __name__ == "__main__":

    #   TODO: validation for the config

    #   This is you have a new model that needs training
    if program_operations["operation"] is "train":

        #   Extracting the data for the dataset that will be used
        chosen_dataset = dataset_settings[training_settings["dataset"]["selected_file"]]

        #   Instantiate Detector and send settings of user for model to use
        training = Training(training_settings, chosen_dataset)

        #   Train model
        model = training.train_model()

        #   Export the model it established in settings
        if program_operations["export_model"]:

            # TODO: create a file saving system that increments by looking into the file and seeing largest number and just
            #       +1 from that one.
            print("WOOO")
            exit()
            joblib.dump(self.model, funcs.autosave_file("Models.file".lower(), 2), compress=1)



    #   This is if you already a model and just want to test it
    elif program_operations["operation"] is "test":

        tester = Testing.validating_model("{0}.file".format(testing_settings["model"].lower()))

        if tester is not None:

            #   Test the model
            tester.test_validation()
            tester.test_recent()
            tester.test_other_datasets()

        else:
            print("The model specified in 'testing_settings' is not valid")

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







