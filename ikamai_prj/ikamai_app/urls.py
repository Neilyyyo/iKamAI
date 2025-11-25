from django.urls import path
from . import views
from django.contrib.auth.views import LoginView

urlpatterns = [
    path('create_account/', views.create_account, name='create_account'),
    path('home/', views.home, name='home'),
    path('login/', views.login_view, name='login'),  
    path('account/', views.account_view, name='account'),
    path('logout/', views.logout_view, name='logout'),
    path('edit_account/', views.edit_account, name='edit_account'),
    path('fsl/', views.fsl, name='fsltotext'),
    path('text/', views.text, name='texttofsl'),
    path('', views.start_page, name='start_page'),
    path("forgot-password/", views.forgot_password_view, name="forgot_password"),

    # Admin Dashboard
    path('admin-dashboard/', views.admin_dashboard, name='admin-dashboard'),
    # path('admin-dashboard/add-video/', views.add_video, name='add_video'),
    path('admin-dashboard/edit-video/<str:video_id>/', views.edit_video, name='edit_video'),
    path('admin-dashboard/delete-video/<str:video_id>/', views.delete_video, name='delete_video'),
    path('admin-dashboard/manage-words/<str:video_id>/', views.manage_words, name='manage_words'),
    path('admin-dashboard/delete-word/<str:word_id>/', views.delete_word, name='delete_word'),

    # User Management
    path('users/', views.user_list, name='user-list'),
    # path('users/add/', views.add_user, name='add-user'),
    # path('users/<str:user_id>/edit/', views.edit_user, name='edit-user'),


    path('change-password/', views.change_password, name='change_password'),

    # Video Management
    path('videos/', views.video_management, name='video-management'),

    # Settings
    path('settings/', views.edit_admin, name='edit_admin'),

    # Analytics
    path('analytics_dashboard/', views.analytics_dashboard, name='analytics_dashboard'),
    path('analytics/export/', views.export_analytics, name='export_analytics'),

    # History
    path('history/', views.history_page, name='history_page'),
    path('save-translation/', views.save_translation, name='save_translation'),
    # path('get-history/', views.get_history, name='get_history'),
    # path('delete-history/', views.delete_history, name='delete_history'),
    #  path("translations/", views.translation_history, name="translation_history"),

    # Sign Language Translation
    path('get-sign-video/', views.get_sign_video, name='get_sign_video'),
    path('translate-phrase/', views.translate_phrase, name='translate_phrase'),
    path('save_sentence/', views.save_sentence, name="save_sentence"),
    path('get_sentencehistory/', views.get_sentencehistory, name='get_sentencehistory'),

    # Prediction
    path('sign/', views.sign, name='sign'), 
    # path('video_feed/', views.video_feed, name='video_feed'),
    path('predict/', views.predict, name='predict'),
    path('apply-suggestion/', views.apply_suggestion, name='apply_suggestion'),
    path('index/', views.detection_page, name='detection_page'),
    path('release_camera/', views.release_camera, name='release_camera'),
    # path('v_feed/', views.v_feed, name='v_feed'),
    path("get_prediction/", views.get_prediction, name="get_prediction"),
    path('reset_prediction/', views.reset_prediction, name='reset_prediction'),

    path("upload/", views.upload_video_view, name="upload_video"),
    path("search/", views.search_video_view, name="search_video"),
    path("stream/<str:file_id>/", views.stream_video_view, name="stream_video"),  # optional proxy
]
