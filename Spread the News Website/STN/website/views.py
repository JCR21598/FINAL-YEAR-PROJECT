from django.shortcuts import render
from django.views import generic
from django.http import HttpResponse

from newsapi import NewsApiClient

from .forms import DetectorForm
from .



""" #   Home View(s)   # """

# Call to retrieve news and display on screen
def home_page(request):

    ###     API

    #   API CALL
    newsapi = NewsApiClient(api_key='9dff3b262af247178cba410205157829')

    #   Type of News requested from API
    top_headlines = newsapi.get_top_headlines(language='en')


    ###     FORM
    if request.method == 'POST':

        # Bound From - populate form instance with data from request
        form = DetectorForm(request.POST)

        # Validate data
        if form.is_valid():

            # Validated data located here
            user_input = form.cleaned_data["user_input"]

            # Next Steps

    else:
        #   Unbound Form - no data being submitted (like when the page is loaded)
        form = DetectorForm()


    #   Designated Template
    template_sent = "website/home.html"

    #   Data sent to Template
    context = {
        "title": "Spread the News",
        "form": form,
        "top_headlines": top_headlines,
    }

    return render(request, template_sent, context)



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