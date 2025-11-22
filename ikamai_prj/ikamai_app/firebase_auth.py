import os
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
from django.conf import settings

# Load Firebase credentials path from settings
firebase_json_path = settings.FIREBASE_CREDENTIALS

# Initialize only once
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json_path)
    firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Auth client (alias)
auth_client = firebase_auth
