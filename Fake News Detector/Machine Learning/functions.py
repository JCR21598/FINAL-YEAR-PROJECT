import pandas as pd
import os
import re


#from cerberus import Validator


#class CustomValidator(Validator):

    #def _validate_inList(self, inList, field, value):

        #   The outline/model of of what the settings dictionary should follow
        # schema = {
        #     "general": {
        #
        #         #   select a file from the list of "available_files"
        #         "selected_file": {"type": "string".lower(), "oneof_allowed": settings['general']['available_files']},
        #
        #         "available_files":{"type": "list"},
        #
        #         #   what happens with NaN values, options: True => consider them, False => Dont consider them
        #         "considering_NaN": False
        #
        #     },
        #     "ML_settings": {
        #
        #         # Select a number between 0 and 1
        #         "validation_set_size": 0.3
        #
        #     },
        #     "testing_settings": {
        #
        #     },
        # }


def check_path_exists(path):

    if os.path.exists(path):
        print("\nPATH EXSISTS")
        print(f"\n  Current Path: {path}")

    else:
        print("PATH DOES NOT EXIST")
        print(f"Invalid Path => {path}")
        exit()


# Read the selected file
def read_dataset(file, available_files, purpose):

    # Current path and check if valid
    path = os.path.dirname(os.path.abspath(__file__))
    check_path_exists(path)

    # Check if file asked by user exsits
    if file.lower() in available_files:

        # Getting the contents of dataset directory chosen from user
        for root, directories, files in os.walk("Project Files\Datasets\{}".format(file.lower())):

            # Find file based off purpose
            requested_file = [x for x in files if purpose in x]

            if len(requested_file) >= 2:
                print(f"There are more than 1 file for {purpose}")
                exit()

            # Remove the list brackets and quotation
            result = ", ".join(requested_file)

            # Need the extension for upcoming process
            file_name , (file_extension) = os.path.splitext(result)

            if file_extension.lower() == ".csv":
                # Reminder: index_col, is to force pandas to not use the first column as index
                csv_df = pd.read_csv("Project Files/Datasets/" + file.lower() + "/" + purpose.lower() + ".csv",
                                     index_col=0)
                return csv_df

            elif file_extension.lower() is ".tsv":

                tsv_df = pd.read_csv("Project Files/Datasets/" + file.lower() + "/" + purpose.lower() + ".tsv",
                                     index_col=0,
                                     sep="\t")
                return tsv_df

            else:

                print("The file extension '{0}' is not valid in {1}".format(file_extension,
                                                                            path + root + file_name))
                exit()

    else:
        print(f"The selected file '{file}' does not exsist")



# We save but split thr label from the Dataframe
def split_label(selected_file, current_df):
    if selected_file.lower() == "kdata":
        label = current_df["type"].copy(deep=True)
        del current_df["type"]
        return current_df, label

    if selected_file.lower() == "liar":
        label = current_df[1].copy(deep=True)
        del current_df[1]
        return current_df, label

    if selected_file.lower() == "github":
        label = current_df["label"].copy(deep=True)
        del current_df["label"]
        return current_df, label


def print_params(name, params):

    print("Parameters for " + name)
    print(params)
    print()



