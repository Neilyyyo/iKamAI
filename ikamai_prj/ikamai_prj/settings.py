from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("SECRET_KEY")
DEBUG = os.getenv("DEBUG") == "False"


ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "127.0.0.1").split(",")# set "example.com,www.example.com"

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    "ikamai_app.apps.IkamaiAppConfig",
    'crispy_forms',
    'crispy_bootstrap5',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware'
]

ROOT_URLCONF = 'ikamai_prj.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'ikamai_app.context_processors.firebase_user',
            ],
        },
    },
]

WSGI_APPLICATION = 'ikamai_prj.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# # Custom User Model
# AUTH_USER_MODEL = 'ikamai_app.CustomUser'

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files
STATIC_ROOT = BASE_DIR / "staticfiles"
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / "static"]


# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Crispy Forms
CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

# Login URLs
# LOGIN_URL = 'login'
# LOGIN_REDIRECT_URL = 'dashboard'
# LOGOUT_REDIRECT_URL = 'login'

FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY")
FIREBASE_CONTINUE_URL = "http://localhost:8000/login"

EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")

import os
import json
import firebase_admin
from firebase_admin import credentials
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# 1. Get the raw string from environment
raw_cred = os.getenv("FIREBASE_CREDENTIALS")

if raw_cred:
    # 2. Convert the string to a Python dictionary
    cred_dict = json.loads(raw_cred)

    # 3. CRITICAL FIX: Even with json.loads, .env sometimes escapes newlines wrong.
    # This line ensures the Private Key is formatted correctly for Firebase.
    cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

    # 4. Initialize Firebase
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    
    print("Success! Connected to project: ikamai-new")
else:
    print("Error: FIREBASE_CREDENTIALS not found in .env file")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B2_APPLICATION_KEY_ID = os.getenv("B2_APPLICATION_KEY_ID")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME")

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")

SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"


