from django.urls import path

# Views
from . import views

urlpatterns = [
    # Home URL(s)
    path('', views.home_page, name="Spread the News"),
    path('input/', views.home_page),

    # About URL(s)
    path('about/', views.about_page),

    # Contact Us URL(s)
    path('contact/', views.contact_page),

    # API URL(s)
    path('api/', views.api_page),

]