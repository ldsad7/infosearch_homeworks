from django.urls import path

from engine import views

urlpatterns = [
    path('', views.main, name='main'),
]
