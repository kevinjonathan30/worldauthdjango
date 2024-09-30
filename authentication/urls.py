from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("get_point_cloud/", views.get_point_cloud, name="get_point_cloud"),
    path("delete_point_cloud/", views.delete_point_cloud, name="delete_point_cloud"),
    path('register/', views.register, name="register"),
    path('authenticate/', views.authenticate, name="authenticate"),
    path('collect/', views.collect, name="collect"),
    path('far_ffr/', views.count_far_ffr, name="count_ffr"),
]