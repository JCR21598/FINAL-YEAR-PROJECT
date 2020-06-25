import logging
import joblib


from detector import Detector
from testing import Testing
from config import program_operations, ML_settings, testing_settings



if __name__ == "__main__":

    '''
    #
    #   Program was developed in Object-Oriented format. This relieved a lot of early developing stages which was a 
    #   program that was hard to track, unsustainable and hard to unmaintainable.
    #
    '''
    #   TODO: In theory should have a validation for the config


    #   This is you have a new model that needs training
    if program_operations["operation"] is "train":

        #   Instantiate Detector and send settings of user for model to use
        detector = Detector(ML_settings)

        #   Train model
        model = detector.train_model()


        #   Export the model it established in settings
        if program_operations["export_model"]:

            # TODO: create a file saving system that increments by looking into the file and seeing largest number and just
            #       +1 from that one.

            joblib.dump(self.model, "Models.file".lower(), compress=1)



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







