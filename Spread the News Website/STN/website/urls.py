from django.urls import path

# Views
from . import views

urlpatterns = [
    path('', views.news_reports, name="Spread the News")
]