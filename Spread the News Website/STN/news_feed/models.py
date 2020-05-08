from django.db import models

# What types of data we doing to be saving:
#   News reports and Users opinion on news report


# News report: headline, summary , body, url, date, author, place, news outlet, image


class NewsReport(models.Model):

    headline = models.CharField(max_length=150)
    news_body = models.CharField(max_length=300)
    news_outlet = models.CharField(max_length=80)
    image = models.CharField