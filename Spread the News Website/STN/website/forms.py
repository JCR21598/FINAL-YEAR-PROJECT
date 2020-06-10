from django import forms

class DetectorForm (forms.Form):

    user_input = forms.URLField(max_length=500, label="", required=True,
                                widget=forms.URLInput(attrs={"id": "detector-field",
                                                               "placeholder":"Verify News",
                                                             }
                                                        ))