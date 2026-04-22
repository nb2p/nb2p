from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("ping", views.ping, name="ping"),
    path("notebook", views.notebook, name="notebook"),
    path("notebook/code/", views.notebook_code, name="notebook_code"),
    path("notebook/cells/", views.notebook_cells, name="notebook_cells"),
    path(
        "notebook/segment-ends/",
        views.notebook_segment_ends,
        name="notebook_segment_ends",
    ),
    path(
        "notebook/segment-ends-v2/",
        views.notebook_segment_ends_v2,
        name="notebook_segment_ends_v2",
    ),
]
