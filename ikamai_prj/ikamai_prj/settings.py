from dotenv import load_dotenv
import os
from pathlib import Path
import json
import firebase_admin
from firebase_admin import credentials

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("SECRET_KEY")
DEBUG = os.getenv("DEBUG") == "True"  # Fixed: should be "True" not "False"

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "ikamai.onrender.com, localhost, 127.0.0.1").split(",")

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

# Firebase Configuration
# Firebase Configuration
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY")

# ✅ ADD THIS LINE so other files can access settings.FIREBASE_CREDENTIALS
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")

# Use different redirect URLs for development vs production
if DEBUG:
    FIREBASE_CONTINUE_URL = "http://localhost:8000/login"
else:
    FIREBASE_CONTINUE_URL = f"https://{ALLOWED_HOSTS[0]}/login"

EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")

# Firebase Admin SDK Initialization with error handling
# ... imports ...

def initialize_firebase():
    # Option A: Check for the standard Env Var (string content)
    raw_cred = os.getenv("FIREBASE_CREDENTIALS")
    
    # Option B: Check for the Render Secret File path
    # Render usually stores secret files in /etc/secrets/
    cred_file_path = "/etc/secrets/ikamai.json"
    
    cred_dict = None

    try:
        # Priority 1: Check if the Secret File exists (Best for Render)
        if os.path.exists(cred_file_path):
            with open(cred_file_path, 'r') as f:
                cred_dict = json.load(f)
                
        # Priority 2: Check if the Env Var string exists (Fallback/Local)
        elif raw_cred:
            cred_dict = json.loads(raw_cred)
            
        else:
            if DEBUG:
                print("Warning: No Firebase credentials found (Env var or File).")
                return
            else:
                raise ValueError("Missing Firebase Credentials!")

        # Fix newlines in private key if necessary
        if "private_key" in cred_dict:
            cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

        # Initialize App
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            print("Firebase Admin SDK initialized successfully")

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in FIREBASE_CREDENTIALS: {e}"
        if DEBUG:
            print(f"Warning: {error_msg}")
        else:
            raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error initializing Firebase: {e}"
        if DEBUG:
            print(f"Warning: {error_msg}")
        else:
            raise

# Initialize Firebase
initialize_firebase()

# Backblaze B2 Configuration
B2_APPLICATION_KEY_ID = os.getenv("B2_APPLICATION_KEY_ID")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME")

SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"