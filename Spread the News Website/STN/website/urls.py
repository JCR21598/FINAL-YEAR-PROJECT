from django.urls import path

# Views
from .views import IndexView

urlpatterns = [
    path('', IndexView.as_view(), name="Spread the News")
]