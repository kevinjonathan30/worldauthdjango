from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('point_cloud/', views.get_point_cloud_data, name="point_cloud"),
    path('register/', views.register, name="register"),
    path('authenticate/', views.authenticate, name="authenticate"),
]