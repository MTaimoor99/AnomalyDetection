from django.urls import path

from . import views

urlpatterns = [
    path("", views.histogram_plot, name="histogram_plot"),
    path("SquareRootTransformation",views.sqrt_transformation,name="sqrt_transformation"),
    path("CubeRootTransformation",views.cube_root_transformation,name="cube_root_transformation"),
    path("LogTransformation",views.log_transformation,name="log_transformation"),
    path("Algorithm",views.algorithm_implementation,name="algorithm_implementation")
]