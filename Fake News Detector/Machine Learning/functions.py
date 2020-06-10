import pandas as pd
import os


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





# Read the selected file
def read_dataset(file, available_files, purpose):

    # Check if the file exists
    print(os.path.abspath(file))
    filename, file_extension = os.path.splitext(str(os.path.abspath(file)))
    print(file_extension)
    exit()

    if os.path.exists(file.lower()):
        print("File path does not exist.")
        exit()

    if file.lower() in available_files:

        # Code for TSV

        # Code for CSV
        csv_df = pd.read_csv("datasets/" + file.lower() + "/" + purpose.lower() + ".csv",
                             index_col=0)  # Reminder: index_col, is to force pandas to not use the first column as index

    else:
        print("file is not available")
        exit()

    return csv_df


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
