from django.urls import path, include
from . import views


urlpatterns = [
    path("index/", views.index, name='index'),
    path("list/", views.list, name='list'),
    path("predict/", views.predict, name='predict'),
    path("info/", views.info, name="info"),
    path("search_stocks/", views.search_stocks, name='search_stocks'),
]