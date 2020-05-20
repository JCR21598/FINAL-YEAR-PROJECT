from django.shortcuts import render
from django.views import generic
from django.http import HttpResponse

from newsapi import NewsApiClient


def news_reports(request):

    #   API
    newsapi = NewsApiClient(api_key='9dff3b262af247178cba410205157829')

    top_headlines = newsapi.get_top_headlines(category='business',
                                              language='en', country='us')

    context = {"top_headlines": top_headlines}



    return render(request, "website/base.html", context)






class DetailView(generic.ListView):

    template_name = "website/templates/website/base.html"








# https://www.youtube.com/watch?time_continue=8&v=QD4GlXtf-WU&feature=emb_logo