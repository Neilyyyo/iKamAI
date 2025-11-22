from django.apps import AppConfig
from django.apps import AppConfig
import firebase_admin
from firebase_admin import credentials, auth, firestore

class IkamaiAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ikamai_app'

    def ready(self):
        # Initialize Firebase only once
        if not firebase_admin._apps:
            cred = credentials.Certificate("AIzaSyDQgZ8pUDkqLDpYID09C2t2p42v2Jkcn0c")
            firebase_admin.initialize_app(cred)

        db = firestore.client()

        DEFAULT_ADMIN_EMAIL = "admin@example.com"
        DEFAULT_ADMIN_PASSWORD = "AdminPass123!"
        DEFAULT_ADMIN_NAME = "System Admin"

        try:
            admin_user = auth.get_user_by_email(DEFAULT_ADMIN_EMAIL)
            print("✅ Default admin already exists:", admin_user.email)
        except auth.UserNotFoundError:
            try:
                admin_user = auth.create_user(
                    email=DEFAULT_ADMIN_EMAIL,
                    password=DEFAULT_ADMIN_PASSWORD,
                    display_name=DEFAULT_ADMIN_NAME,
                )
                print("🎉 Created default admin in Firebase:", admin_user.uid)

                db.collection("users").document(admin_user.uid).set({
                    "uid": admin_user.uid,
                    "email": DEFAULT_ADMIN_EMAIL,
                    "display_name": DEFAULT_ADMIN_NAME,
                    "role": "admin",
                    "created_at": firestore.SERVER_TIMESTAMP
                })
                print("✅ Default admin added to Firestore.")
            except Exception as e:
                print("❌ Failed to create default admin:", e)

