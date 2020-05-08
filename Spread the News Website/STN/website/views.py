from django.shortcuts import render
from django.views import generic
from django.http import HttpResponse


from STN.news_feed.models import NewsReport


class IndexView(generic.ListView):

    template_name = "website/templates/website/index.html"

    def get_newsfeed(self):

        news_feed = NewsReport.objects.all()
        context = news_feed

        return HttpResponse()

    def get_queryset(self):
            pass


