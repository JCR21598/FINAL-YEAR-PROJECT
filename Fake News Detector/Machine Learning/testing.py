
# Python Libraries
import os

# SKLEARN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualisation
import matplotlib

import detector
import functions as funcs
import decorators as dec


class Testing:

    def __init__(self, model):
        pass


    @classmethod
    @dec.print_simple_divider
    def validating_model(cls, model):

        model_path = os.path.dirname(os.path.realpath(__file__ )) + "\Project Files\Train Results\Models"
        funcs.check_path_exists(model_path)

        # Check if any of the model files mathches users desired model to test on
        file = "{0}\{1}".format(model_path, model)
        funcs.check_path_exists(file)

        file_existence = os.path.isfile()



        # If model exsists then create an instance
        if file_exsistence:
            return cls(model)
        # Otherwise just return None
        return None


    ###     Testing Methods

    #   Testing with Validation Dataset
    def test_validation(self):

        validation_predicted = model.predict(self.X_test)

        print(f"\n\nAccuracy with {self.selected_file.lower()} VALIDATION dataset:",
              accuracy_score(self .y_test, validation_predicted))

        print(f"\n\nClassification Report for {self.selected_file.lower()} VALIDATION dataset:")
        print(classification_report(self.y_test, validation_predicted))

        print(f"\n\nConfusion Matrix for {self.selected_file.lower()} VALIDATION dataset:")
        print(confusion_matrix(self.y_test, validation_predicted))



    #   Testing with Recent Data
    def test_recent(self):
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

        test_label = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0])

        test_df = pd.Series(test)
        test_label_df = pd.Series(test_label)

        real_data_predicted = model.predict(test_df)

        print(f"\n\nAccuracy with REAL dataset:",
              accuracy_score(test_label_df, real_data_predicted))

        print(f"\n\nClassification Report for REAL dataset:")
        print(classification_report(test_label_df, real_data_predicted))

        print(f"\n\nConfusion Matrix for REAL dataset:")
        print(confusion_matrix(test_label_df, real_data_predicted))



    #   Testing from data of other Datasets
    def test_other_datasets(self):
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

        test_label2 = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1])

        test_df2 = pd.Series(test)
        test_label_df2 = pd.Series(test_label)

        real_data_predicted2 = model.predict(test_df2)

        print(f"\n\nAccuracy with OTHER dataset:",
              accuracy_score(test_label_df2, real_data_predicted2))

        print(f"\n\nClassification Report for OTHER dataset:")
        print(classification_report(test_label_df2, real_data_predicted2))

        print(f"\n\nConfusion Matrix for OTHER dataset:")
        print(confusion_matrix(test_label_df2, real_data_predicted2))
