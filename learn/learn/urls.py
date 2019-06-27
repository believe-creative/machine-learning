
from django.contrib import admin
from django.urls import path
from learn.views import get_string,imageView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('get_string/', get_string),
    path('image_view/', imageView.as_view()),
]
