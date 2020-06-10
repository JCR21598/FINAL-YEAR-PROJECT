from django.shortcuts import render
from django.views import generic
from django.http import HttpResponse, JsonResponse

from newsapi import NewsApiClient
from newspaper import Article

from .forms import DetectorForm



""" #   Home View(s)   # """

# Call to retrieve news and display on screen
def home_page(request):
    ###     FORM
    #   Unbound Form
    form = DetectorForm()

    ###     API
    #   API CALL
    newsapi = NewsApiClient(api_key='9dff3b262af247178cba410205157829')

    #   Type of News requested from API
    top_headlines = newsapi.get_top_headlines(language='en')


    ###     Prep for Response
    #   Designated Template
    template_sent = "website/home.html"

    #   Data sent to Template
    context = {
        "title": "Spread the News",
        "form": form,
        "top_headlines": top_headlines,
    }

    return render(request, template_sent, context)


#   AJAX
def url_prediction(request):

    #   Data already arrived validated and not-blank
    if request.method == 'POST' and request.is_ajax():

        json_response = []

        #   Grabbing submitted URLs and store in list
        user_input = request.POST["url_input"]
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


            temp_json = {
                "url": url,
                "article_title": article_title,
                "article_text": article_text,
            }

            #   Append the urls - this is the data that is sent to JavaScript file to display
            json_response.append(temp_json)


        print(json_response)


    else:
        print("ERROR -  Not POST and/or AJAX")

    return JsonResponse(json_response, status=200)


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



""" #   Contact Us View(s)   # """
def contact_page(request):

    #   Designated Template
    template_sent = "website/contact.html"

    #   Data sent to Template
    context = {
        "title": "Contact Us",
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