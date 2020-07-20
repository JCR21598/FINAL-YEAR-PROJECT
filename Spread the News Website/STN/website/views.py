
#   Python Imports
import os

#   Django Imports
from django.views.generic import ListView
from django.shortcuts import render
from django.views import generic
from django.http import HttpResponse, JsonResponse, Http404, HttpResponseNotFound

#   Other Third-Party resources
from newsapi import NewsApiClient
from newspaper import Article
import sklearn
import joblib

#   Personal modules
from .forms import DetectorForm


""" #   Home View(s)   # """

# Call to retrieve news and display on screen
def home_page(request):
    ###     FORM
    #   Unbound Form
    form = DetectorForm()

    ###     Gather News Feed


    # #   API CALL
    # newsapi = NewsApiClient(api_key='9dff3b262af247178cba410205157829')
    #
    # #   Type of News requested from API
    # top_headlines = newsapi.get_top_headlines(language='en')


    ###     Prep for Response
    #   Designated Template
    template_sent = "website/home.html"

    #   Data sent to Template
    context = {
        "title": "Spread the News",
        "form": form,
        #"top_headlines": top_headlines,
    }

    return render(request, template_sent, context)




    # TODO: consider the fact that by allowing multiple URLS then only one needs to be valid for it to be submitted
    #               - A solution can be that you dont make it a required field and use the valid function for each URL wihtin the for loop in the list with try...except...

#   AJAX
def url_prediction(request):

    json_response = []

    #   Data already arrived validated and not-blank
    if request.method == 'POST' and request.is_ajax():

        #   Loading in classifier
        CURRENT_DIR = os.path.dirname(__file__)
        model_file = os.path.join(CURRENT_DIR, 'ml_models/model1.file')

        model = joblib.load(model_file)

        #   Grabbing submitted URLs and store in list
        user_input = request.POST["user_input"]
        urls = user_input.split(" ")

        print(urls)

        for url in urls:

            #   Obtain news from url
            article = Article(url)
            article.download()

            #   Extract the contents of the Newspaper
            article.parse()

            #   Getting content that is of interest
            article_title = article.title
            article_text = article.text

            #   For the predictor we need the value in an iterable format - even if it just a value
            article_title_list = [article_title]

            #   Returns a list with prediction
            prediction_np = model.predict(article_title_list)

            #   However, cannot use numpy array as it causes more difficulties with JavaScript
            prediction = str(prediction_np).lstrip('[').rstrip(']')
            print(prediction)


            if prediction == "0":
                prediction = "Fake News"

            elif prediction == "1":
                prediction= "Realiable News"

            else:
                prediction = "Internal Error"


            #   Results sent to JavaScript from each news report
            temp_json = {
                "url": url,
                "article_title": article_title,
                "article_text": article_text,
                "prediction": prediction,
            }

            #   Append the urls - this is the data that is sent to JavaScript file to display
            json_response.append(temp_json)

        print(json_response)

        return JsonResponse(json_response, status=200, safe=False)

    else:
        print("ERROR -  Not POST and/or AJAX")

        # TODO: Create template for this error

        return HttpResponseNotFound()




class DetailView(generic.ListView):

    template_name = "website/templates/website/second.html"



""" #   About View(s)   # """
def about_page(request):

    #   Designated Template
    template_sent = "website/about.html"

    #   Data sent to Template
    context = {
        "title": "About",
    }

    return render(request, template_sent, context)


""" #   API View(s)   # """
def api_page(request):

    #   Designated Template
    template_sent = "website/api.html"

    #   Data sent to Template
    context = {
        "title": "API",
    }

    return render(request, template_sent, context)



# https://www.youtube.com/watch?time_continue=8&v=QD4GlXtf-WU&feature=emb_logo