from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("get_point_cloud/", views.get_point_cloud, name="get_point_cloud"),
    path("delete_point_cloud/", views.delete_point_cloud, name="delete_point_cloud"),
    path('register/', views.register, name="register"),
    path('authenticate/', views.authenticate, name="authenticate"),
    path('authenticate_two/', views.authenticate_two, name="authenticate_two"),
    path('save_data/', views.save_data, name="save_data"),
]