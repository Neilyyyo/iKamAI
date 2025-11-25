from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
import json
import csv
from .models import User, History, SignLanguageVideo, VideoWord, Sentence, SignLanguageHistory
from firebase_admin import auth as firebase_auth, storage
from django.views.decorators.csrf import csrf_exempt
from functools import wraps
from django.core.paginator import Paginator
from datetime import datetime, timedelta
import re
from .predictor import SignPredictor
from django.views.decorators import gzip
# views.py
import firebase_admin
from django.utils import timezone
from firebase_admin import auth, credentials, firestore
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse  # ADD THIS
from django.utils.timezone import now
from functools import wraps
import os
from django.conf import settings
import requests
import json
import threading
from firebase_admin import auth as firebase_auth

from django.views.decorators.http import require_POST
from datetime import datetime
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
# from google.cloud.firestore_v1 import DocumentSnapshot
from .utils import WordPredictor
from django.conf import settings
import firebase_admin
from firebase_admin import credentials




db = firestore.client()

class FirestoreJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Firestore datetime objects"""
    def default(self, obj):
        # Handle Firestore timestamp objects
        if hasattr(obj, '_seconds') and hasattr(obj, '_nanoseconds'):
            try:
                timestamp = obj._seconds + (obj._nanoseconds / 1e9)
                dt = datetime.fromtimestamp(timestamp)
                return dt.isoformat()
            except:
                return str(obj)
        
        # Handle regular datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle other non-serializable objects
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
    
    def convert_firestore_dates(self, obj):
        """Recursively convert Firestore dates in nested structures"""
        if isinstance(obj, dict):
            return {k: self.convert_firestore_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_firestore_dates(item) for item in obj]
        elif hasattr(obj, '_seconds') and hasattr(obj, '_nanoseconds'):
            return datetime.fromtimestamp(obj._seconds + obj._nanoseconds / 1e9).isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return obj

def custom_json_response(data, status=200):
    """Custom JSON response that handles Firestore objects"""
    return JsonResponse(data, encoder=FirestoreJSONEncoder, safe=False, status=status)

# Initialize Firebase Admin SDK
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate(settings.FIREBASE_CONFIG)
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# Firebase REST API authentication - SINGLE VERSION
def firebase_sign_in(email, password, api_key):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    data = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        result = response.json()
        
        if response.status_code == 200:
            return result
        else:
            # Handle specific Firebase error messages
            error_message = result.get('error', {}).get('message', 'Unknown error')
            if 'INVALID_PASSWORD' in error_message:
                raise Exception('Invalid password')
            elif 'EMAIL_NOT_FOUND' in error_message:
                raise Exception('Email not found')
            elif 'USER_DISABLED' in error_message:
                raise Exception('User account has been disabled')
            elif 'TOO_MANY_ATTEMPTS_TRY_LATER' in error_message:
                raise Exception('Too many failed attempts. Please try again later')
            else:
                raise Exception(error_message)
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        if "Network error" in str(e) or "Authentication failed" in str(e):
            raise e
        else:
            raise Exception(f"Authentication failed: {str(e)}")

# SINGLE LOGIN VIEW - IMPROVED VERSION
def login_view(request):
    # Redirect if already logged in
    if 'uid' in request.session:
        print("User already logged in, redirecting to home")
        return redirect('home')
        
    if request.method == 'POST':
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        
        print(f"Login attempt for email: {email}")  # Debug
        
        # Basic validation
        if not email or not password:
            messages.error(request, 'Both email and password are required.')
            return render(request, 'login.html')
        
        # Get your Firebase API key

        FIREBASE_API_KEY = settings.FIREBASE_WEB_API_KEY

        try:
            # Authenticate with Firebase using REST API
            print("Attempting Firebase authentication...")
            result = firebase_sign_in(email, password, FIREBASE_API_KEY)
            print(f"Firebase auth successful for user: {result['localId']}")
            
            # Verify the user exists in Firebase Auth (additional security check)
            try:
                firebase_user = auth.get_user(result['localId'])
                print(f"User verified in Firebase Auth: {firebase_user.email}")
            except auth.UserNotFoundError:
                messages.error(request, 'User not found in authentication system.')
                return render(request, 'login.html')
            
            # Store user ID and token in session
            request.session['uid'] = result['localId']
            request.session['id_token'] = result['idToken']
            request.session['refresh_token'] = result['refreshToken']
            print(f"Session data stored for user: {result['localId']}")
            
            # Get user data from Firestore
            try:
                user_doc = db.collection('users').document(result['localId']).get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    request.session['user_data'] = user_data
                    print(f"User data loaded from Firestore: {user_data.get('username', email)}")
                else:
                    # User exists in Firebase Auth but not in Firestore
                    print(f"Warning: User {result['localId']} exists in Auth but not in Firestore")
                    # Create basic user data in Firestore
                    basic_user_data = {
                        'uid': result['localId'],
                        'email': firebase_user.email,
                        'display_name': firebase_user.display_name or '',
                        'created_at': firestore.SERVER_TIMESTAMP
                    }
                    db.collection('users').document(result['localId']).set(basic_user_data)
                    request.session['user_data'] = basic_user_data
                    print("Created basic user data in Firestore")
                    
            except Exception as firestore_error:
                print(f"Firestore error: {str(firestore_error)}")
                # Continue login even if Firestore fails
                request.session['user_data'] = {
                    'uid': result['localId'],
                    'email': firebase_user.email,
                    'display_name': firebase_user.display_name or ''
                }
                
            messages.success(request, 'Logged in successfully!')
            print("Login successful, redirecting to home...")
            
            # Redirect to intended page or home
            next_page = request.GET.get('next', 'home')
            return redirect(next_page)
            
        except Exception as e:
            error_message = str(e)
            print(f"Login error: {error_message}")  # Debug log

            # 🔹 Map Firebase errors to user-friendly messages
            if "auth/invalid-credential" in error_message:
                friendly_message = "Invalid email or password. Please try again."
            elif "auth/user-not-found" in error_message:
                friendly_message = "No account found with this email."
            elif "auth/wrong-password" in error_message:
                friendly_message = "Incorrect password. Please try again."
            else:
                friendly_message = "An error occurred while signing in. Please try again."

            messages.error(request, friendly_message)
            return render(request, 'login.html')


def logout_view(request):
    """Logout function to clear session"""
    print("User logging out...")
    request.session.flush()
    messages.success(request, 'Logged out successfully!')
    return redirect('login')

# Home view - ADD THIS IF YOU DON'T HAVE IT
def home(request):
    """Home page view"""
    print(f"Home view accessed. Session UID: {request.session.get('uid', 'None')}")
    
    if 'uid' not in request.session:
        print("No UID in session, redirecting to login")
        messages.error(request, 'Please log in to access this page.')
        return redirect('login')
    
    user_data = request.session.get('user_data', {})
    print(f"User data in session: {user_data}")
    
    context = {
        'user_data': user_data
    }
    return render(request, 'home.html', context)

# Firebase authentication decorator
def firebase_login_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if 'uid' not in request.session:
            return redirect('/login/')
        return view_func(request, *args, **kwargs)
    return wrapper

# Helper function to check if username exists
def username_exists(username):
    try:
        users_ref = db.collection('users')
        query = users_ref.where('username', '==', username).limit(1)
        return len(query.get()) > 0
    except Exception as e:
        print(f"Error checking username: {e}")
        return False

# Helper function to check if email exists
def email_exists(email):
    try:
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1)
        return len(query.get()) > 0
    except Exception as e:
        print(f"Error checking email: {e}")
        return False

def create_account(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        username = request.POST.get('username')
        password = request.POST.get('password')

        print(f"Account creation attempt for email: {email}")

        # Validate input
        if not all([first_name, last_name, email, username, password]):
            messages.error(request, 'All fields are required.')
            return render(request, 'create_account.html')

        try:
            # Check if username already exists
            if username_exists(username):
                messages.error(request, 'Username already taken. Please choose a different one.')
                return render(request, 'create_account.html')
                
            # Check if email already exists
            if email_exists(email):
                messages.error(request, 'Email already registered. Please use a different email.')
                return render(request, 'create_account.html')

            # Create user in Firebase Auth - USING FIREBASE AUTH
            try:
                user = firebase_auth.create_user(
                    email=email,
                    password=password,
                    display_name=f"{first_name} {last_name}"
                )
                print(f"User created in Firebase Auth: {user.uid}")
            except Exception as e:
                print(f"Firebase Auth creation error: {e}")
                messages.error(request, f'Error creating account: {str(e)}')
                return render(request, 'create_account.html')

            # Create user in Firestore
            try:
                user_data = {
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': email,
                    'username': username,
                    'created_at': firestore.SERVER_TIMESTAMP
                }
                db.collection('users').document(user.uid).set(user_data)
                print(f"User data saved to Firestore: {user.uid}")
                
                messages.success(request, 'Account created successfully! You can now log in.')
                return redirect('login')
            except Exception as e:
                print(f"Firestore creation error: {e}")
                # If Firestore creation fails, delete the user from Firebase Auth
                try:
                    firebase_auth.delete_user(user.uid)
                    print(f"Deleted user from Firebase Auth due to Firestore error: {user.uid}")
                except:
                    pass
                messages.error(request, f'Error saving user data: {str(e)}')
                return render(request, 'create_account.html')

        except Exception as e:
            print(f"Unexpected error in create_account: {e}")
            messages.error(request, f'Unexpected error: {str(e)}')
            return render(request, 'create_account.html')

    return render(request, 'create_account.html')
import requests
from django.shortcuts import redirect, render
from django.contrib import messages
from django.views.decorators.http import require_http_methods

import requests
from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
import firebase_admin
from firebase_admin import credentials, firestore
# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(settings.FIREBASE_WEB_API_KEY)  # JSON path stored in settings
    firebase_admin.initialize_app(cred)

db = firestore.client()


@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        try:
            # Firebase Auth REST API
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={settings.FIREBASE_WEB_API_KEY}"
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            res = requests.post(url, json=payload)
            data = res.json()

            if "error" in data:
                messages.error(request, "Invalid email or password.")
                return render(request, "login.html")

            # Login success
            uid = data["localId"]
            id_token = data["idToken"]

            # Get user role from Firestore
            user_doc = db.collection("users").document(uid).get()
            role = "user"  # default
            if user_doc.exists:
                user_data = user_doc.to_dict()
                role = user_data.get("role", "user")  # only take role field

            # Store only primitive values in session
            request.session["uid"] = uid
            request.session["email"] = email
            request.session["id_token"] = id_token
            request.session["role"] = role

            print(f"✅ Firebase login successful for {email} (UID: {uid}, Role: {role})")

            # Redirect based on role
            if role == "admin":
                return redirect("/admin-dashboard/")
            else:
                return redirect("/home/")

        except Exception as e:
            print("Login error:", str(e))
            messages.error(request, "Something went wrong.")
            return render(request, "login.html")

    return render(request, "login.html")

@firebase_login_required
def home(request):
    return render(request, 'home.html')

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages

@firebase_login_required
def account_view(request):
    try:
        uid = request.session.get("uid")
        if not uid:
            messages.error(request, "No user session found. Please log in.")
            return redirect("login")

        user_doc = db.collection('users').document(uid).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()

            # Ensure all fields exist, even if blank
            context = {
                "first_name": user_data.get("first_name", ""),
                "last_name": user_data.get("last_name", ""),
                "email": user_data.get("email", ""),
                "username": user_data.get("username", ""),
                # don’t send password directly
            }

            return render(request, "account.html", {"user": context})
        else:
            messages.error(request, "User data not found.")
            return redirect("home")
    except Exception as e:
        messages.error(request, f"Error retrieving account: {str(e)}")
        return redirect("home")


@firebase_login_required
def fsl(request):
    return render(request, 'fsltotext.html')

@firebase_login_required
def text(request):
    return render(request, 'texttofsl.html')

def logout_view(request):
    if 'uid' in request.session:
        del request.session['uid']
    if 'user_data' in request.session:
        del request.session['user_data']
    return redirect('start_page')

def start_page(request):
    return render(request, 'start_page.html')

from dotenv import load_dotenv
load_dotenv()

@firebase_login_required
def account_view(request):
    # 2. Get the specific keys needed for the frontend SDK
    context = {
        'api_key': os.getenv('VITE_FIREBASE_API_KEY'),
        'auth_domain': os.getenv('VITE_FIREBASE_AUTH_DOMAIN'),
        'project_id': os.getenv('VITE_FIREBASE_PROJECT_ID'),
        'storage_bucket': os.getenv('VITE_FIREBASE_STORAGE_BUCKET'),
        'messaging_sender_id': os.getenv('VITE_FIREBASE_MESSAGING_SENDER_ID'),
        'app_id': os.getenv('VITE_FIREBASE_APP_ID'),
        'measurement_id': os.getenv('VITE_FIREBASE_MEASUREMENT_ID'),
    }
    
    # 3. Pass 'context' to the template
    return render(request, 'account.html', context)

@firebase_login_required
def edit_account(request):
    if request.method == "POST":
        try:
            user_ref = db.collection('users').document(request.session['uid'])
            
            # Update user data in Firestore
            user_ref.update({
                'first_name': request.POST.get('first_name'),
                'last_name': request.POST.get('last_name'),
                'email': request.POST.get('email'),
                'username': request.POST.get('username')
            })
            
            # Update password if provided
            new_password = request.POST.get('password')
            if new_password:
                auth.update_user(request.session['uid'], password=new_password)

            messages.success(request, "Your information has been updated successfully!")
            return redirect('account')
        except Exception as e:
            messages.error(request, f"Error updating account: {str(e)}")
            return redirect('account')

    # For GET requests, show the account page
    return redirect('account')

@csrf_exempt
@firebase_login_required
def save_translation(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            History.collection.create(
                user=f'users/{request.user.uid}',
                input_text=data.get('text'),
                video_path=data.get('video_path')
            )
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import datetime
from firebase_admin import firestore
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import datetime
from firebase_admin import firestore

db = firestore.client()

@csrf_exempt
def save_translation_words(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get("text")
            user_id = data.get("user_id")  # get UID from frontend

            if text and user_id:
                db.collection("translations").add({
                    "text": text,
                    "timestamp": datetime.utcnow(),
                    "user_id": user_id,
                    "type": "words"
                })
                return JsonResponse({"status": "success", "message": "Translation saved"})
            else:
                return JsonResponse({"status": "error", "message": "Missing text or user_id"}, status=400)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Invalid request"}, status=400)

# views.py
from django.shortcuts import render
from django.utils.dateparse import parse_datetime
from .context_processors import get_translation_history

# @firebase_login_required
# def get_history(request):
#     start_date = request.GET.get("start_date")
#     end_date = request.GET.get("end_date")

#     if start_date and end_date:
#         history = get_translation_history(parse_datetime(start_date), parse_datetime(end_date))
#     else:
#         history = get_translation_history()

#     return render(request, "history.html", {"history": history})

# @csrf_exempt
# @firebase_login_required
# def delete_history(request):
#     if request.method == 'POST':
#         try:
#             history = History.collection.filter('user', '==', f'users/{request.user.uid}').fetch()
#             for item in history:
#                 item.delete()
#             return JsonResponse({'status': 'success'})
#         except Exception as e:
#             return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
#     return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

# @firebase_login_required
# def add_video(request):
#     if request.method == 'POST':
#         title = request.POST.get('title')
#         video_file = request.FILES.get('video_file')
#         words = request.POST.getlist('words')

#         try:
#             bucket = storage.bucket()
#             file_path = f'sign_videos/{video_file.name}'
#             blob = bucket.blob(file_path)
#             blob.upload_from_file(video_file)
#             video_url = blob.public_url

#             video = SignLanguageVideo.collection.create(
#                 title=title,
#                 video_file=video_url,
#                 video_path=file_path
#             )

#             for word in words:
#                 if word:
#                     VideoWord.collection.create(
#                         video=video.key,
#                         word=word.lower()
#                     )
            
#             messages.success(request, f'Video "{title}" added successfully!')
#             return redirect('admin-dashboard')
#         except Exception as e:
#             messages.error(request, f'Error adding video: {e}')
#             return redirect('add_video')

#     return render(request, 'add_video.html')

@firebase_login_required
def edit_video(request, video_id):
    video = SignLanguageVideo.collection.get(f'sign_language_videos/{video_id}')
    if request.method == 'POST':
        video.title = request.POST.get('title')
        video.save()
        messages.success(request, f'Video "{video.title}" updated successfully!')
        return redirect('manage_words', video_id=video.id)
    return render(request, 'edit_video.html', {'video': video})


@firebase_login_required
def manage_words(request, video_id):
    video = SignLanguageVideo.collection.get(f'sign_language_videos/{video_id}')
    words = VideoWord.collection.filter('video', '==', video.key).fetch()
    
    if request.method == 'POST':
        word = request.POST.get('word')
        if word:
            VideoWord.collection.create(
                video=video.key,
                word=word.lower()
            )
            messages.success(request, f'Word "{word}" added to video!')
            return redirect('manage_words', video_id=video.id)
    
    return render(request, 'manage_words.html', {'video': video, 'words': words})


@firebase_login_required
def delete_video(request, video_id):
    if request.method == "POST":
        docs = db.collection("videos").where("file_id", "==", video_id).get()
        for doc in docs:
            db.collection("videos").document(doc.id).delete()
    return redirect("video-management")


@firebase_login_required
def delete_word(request, word_id):
    if request.method == "POST":
        word = VideoWord.collection.get(f'video_words/{word_id}')
        video_id = word.video.id
        word.delete()
        messages.success(request, f'Word "{word.word}" deleted successfully.')
        return redirect('manage_words', video_id=video_id)
    return redirect('manage_words', video_id=word.video.id)

from firebase_admin import auth as firebase_auth

@firebase_login_required
def admin_dashboard(request):
    # ✅ Totals
    total_users = len(list(db.collection("users").stream()))
    total_videos = len(list(db.collection("videos").stream()))
    total_translations = len(list(db.collection("translations").stream()))
    total_searches = len(list(db.collection("search_history").stream()))  # New

    # ✅ Recent users (from users collection)
    recent_users = (
        db.collection("users")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(5)
        .stream()
    )
    recent_users = [doc.to_dict() for doc in recent_users]

    # ✅ Recent videos
    recent_videos = [doc.to_dict() for doc in db.collection("videos").limit(5).stream()]

    # ✅ All translations (not just recent)
    translations_docs = (
        db.collection("translations")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .stream()
    )

    translations = []
    for doc in translations_docs:
        data = doc.to_dict()
        user_id = data.get("user_id")

        email = "Unknown"
        if user_id:
            try:
                # ✅ Fetch email directly from Firebase Authentication
                user_record = firebase_auth.get_user(user_id)
                email = user_record.email
            except Exception as e:
                print(f"⚠️ Auth lookup failed for {user_id}: {e}")

        data["email"] = email
        data["type"] = "Translation"  # Add type for display
        translations.append(data)

    # ✅ All search history (not just recent)
    search_docs = (
        db.collection("search_history")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .stream()
    )

    searches = []
    for doc in search_docs:
        data = doc.to_dict()
        user_id = data.get("user_id")

        email = "Unknown"
        if user_id:
            try:
                # ✅ Fetch email directly from Firebase Authentication
                user_record = firebase_auth.get_user(user_id)
                email = user_record.email
            except Exception as e:
                print(f"⚠️ Auth lookup failed for {user_id}: {e}")

        data["email"] = email
        data["type"] = "Search"  # Add type for display
        searches.append(data)

    # ✅ Take top 5 for dashboard preview
    recent_translations = translations[:5]
    recent_searches = searches[:5]  # New

    # ✅ Combine activities for activity feed
    all_activities = translations + searches
    all_activities.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    recent_activities = all_activities[:10]  # Top 10 recent activities

    context = {
        "total_users": total_users,
        "total_videos": total_videos,
        "total_translations": total_translations,
        "total_searches": total_searches,  # New
        "recent_users": recent_users,
        "recent_videos": recent_videos,
        "recent_translations": recent_translations,
        "recent_searches": recent_searches,  # New
        "recent_activities": recent_activities,  # New
        "translations": translations,
        "searches": searches,  # New
    }

    return render(request, "admin_dashboard.html", context)

def user_management(request):
    users_list = list(User.collection.fetch())
    paginator = Paginator(users_list, 10)
    page_number = request.GET.get('page')
    users = paginator.get_page(page_number)
    
    return render(request, 'user_management.html', {'users': users})

# @firebase_login_required
# def add_user(request):
#     if request.method == 'POST':
#         email = request.POST.get('email')
#         password = request.POST.get('password')
#         username = request.POST.get('username')
#         try:
#             user = auth.create_user(email=email, password=password, display_name=username)
#             User.collection.create(id=user.uid, email=email, username=username)
#             messages.success(request, f"User {username} created successfully")
#         except Exception as e:
#             messages.error(request, f"Error creating user: {str(e)}")
#     return redirect('user-management')

# @firebase_login_required
# def edit_user(request, user_id):
#     if request.method == 'POST':
#         try:
#             auth.update_user(user_id, email=request.POST.get('email'), display_name=request.POST.get('username'))
#             user = User.collection.get(f'users/{user_id}')
#             user.email = request.POST.get('email')
#             user.username = request.POST.get('username')
#             user.save()
#             messages.success(request, f"User {user.username} updated successfully")
#         except Exception as e:
#             messages.error(request, f"Error updating user: {str(e)}")
#     return redirect('user-management')

from django.http import JsonResponse
from firebase_admin import auth as firebase_auth

def user_list(request):
    firebase_users = []  
    # however you fetch users from Firebase...
    for u in auth.list_users().users:
        firebase_users.append({
            "uid": u.uid,   # 👈 required for delete link
            "first_name": u.display_name.split()[0] if u.display_name else "",
            "last_name": u.display_name.split()[-1] if u.display_name else "",
            "email": u.email,
            "username": u.display_name or "",
            "date_created": u.user_metadata.creation_timestamp,
        })
    return render(request, "user_list.html", {"users": firebase_users})



# views.py
from django.shortcuts import render
from django.core.paginator import Paginator
from .firebase_auth import db  # ✅ Firestore client

from django.shortcuts import render
from firebase_admin import firestore

db = firestore.client()

def video_management(request):
    search_query = request.GET.get("search", "").strip()
    current_sort = request.GET.get("sort", "-file_id")  # default newest first

    # Fetch videos from Firestore
    videos_ref = db.collection("videos")
    docs = videos_ref.stream()

    videos = []
    for doc in docs:
        data = doc.to_dict()
        data["file_id"] = data.get("file_id", "")
        data["file_name"] = data.get("file_name", "")
        data["title"] = data.get("title", "")
        data["url"] = data.get("url", "")
        videos.append(data)

    # 🔎 Apply search filter
    if search_query:
        videos = [
            v for v in videos
            if search_query.lower() in v.get("title", "").lower()
               or search_query.lower() in v.get("file_name", "").lower()
        ]

    # 🔀 Apply sorting
    reverse = False
    sort_key = current_sort

    if current_sort.startswith("-"):
        reverse = True
        sort_key = current_sort[1:]  # strip the "-"

    try:
        videos.sort(key=lambda v: v.get(sort_key, ""), reverse=reverse)
    except KeyError:
        pass  # if field missing, skip sorting

    context = {
        "videos": videos,
        "search_query": search_query,
        "current_sort": current_sort,
    }
    return render(request, "video_management.html", context)

from firebase_admin import firestore, auth
from django.contrib import messages
from django.shortcuts import render, redirect

db = firestore.client()

@firebase_login_required
def edit_admin(request):
    if request.method == 'POST':
        uid = request.user.uid
        user_ref = db.collection('users').document(uid)
        user_doc = user_ref.get()

        if not user_doc.exists:
            messages.error(request, 'User not found!')
            return redirect('edit_admin')

        form_type = request.POST.get('form_type')

        if form_type == 'personal_info':
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')

            user_ref.update({
                'first_name': first_name,
                'last_name': last_name,
            })
            messages.success(request, 'Personal information updated successfully!')

        elif form_type == 'email':
            new_email = request.POST.get('email')
            current_email = user_doc.to_dict().get('email')

            if new_email and current_email != new_email:
                # Update Firebase Auth
                auth.update_user(uid, email=new_email)
                # Update Firestore
                user_ref.update({'email': new_email})
                messages.success(request, 'Email address updated successfully!')
            else:
                messages.info(request, 'Email address unchanged')

        elif form_type == 'password':
            new_password = request.POST.get('new_password')
            if new_password:
                auth.update_user(uid, password=new_password)
                messages.success(request, 'Password updated successfully!')

        return redirect('edit_admin')

    return render(request, 'admin_settings.html')

# views.p
import json
from django.shortcuts import render
from collections import Counter
from datetime import datetime
from .firebase_auth import db


import firebase_admin
from firebase_admin import credentials, firestore
from django.conf import settings
from collections import defaultdict, Counter
from datetime import datetime

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(settings.FIREBASE_CONFIG)
    firebase_admin.initialize_app(cred)
import json
from django.shortcuts import render
from firebase_admin import firestore

db = firestore.client()

def analytics_dashboard(request):
    # --- Count translations per user_id ---
    translations_ref = db.collection('translations')
    translations = translations_ref.stream()

    user_activity_count = {}
    for t in translations:
        data = t.to_dict()
        user_id = data.get("user_id")
        if user_id:
            user_activity_count[user_id] = user_activity_count.get(user_id, 0) + 1

    # --- Count search history per user_id ---
    history_ref = db.collection('search_history')
    history = history_ref.stream()

    for h in history:
        data = h.to_dict()
        user_id = data.get("user_id")
        if user_id:
            user_activity_count[user_id] = user_activity_count.get(user_id, 0) + 1

    # --- Fetch usernames once ---
    users_ref = db.collection('users')
    users = {}
    for u in users_ref.stream():
        data = u.to_dict()
        uid = u.id
        username = data.get("username") or data.get("email") or uid
        users[uid] = username

    # --- Build top_users with username ---
    top_users = [
        {"username": users.get(uid, uid), "count": count}
        for uid, count in sorted(user_activity_count.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    # --- User signups per month ---
    user_signups = {}
    for u in users_ref.stream():
        data = u.to_dict()
        created_at = data.get("created at") or data.get("created_at")
        if created_at:
            if hasattr(created_at, "to_datetime"):
                dt = created_at.to_datetime()
            else:
                dt = created_at
            month = dt.strftime("%Y-%m")
            user_signups[month] = user_signups.get(month, 0) + 1

    signup_data = [{"month": m, "count": c} for m, c in sorted(user_signups.items())]

    # --- Popular translations ---
    query_count = {}
    history_ref = db.collection('search_history')
    history = history_ref.stream()

    for h in history:
        data = h.to_dict()
        query = data.get("query")
        if query:
            query_count[query] = query_count.get(query, 0) + 1

    popular_queries = [
        {"query": q, "count": c}
        for q, c in sorted(query_count.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    context = {
        "top_users": json.dumps(top_users),
        "signup_data": json.dumps(signup_data),
        "popular_queries": json.dumps(popular_queries),
    }

    return render(request, "analytics.html", context)



def export_analytics(request):
    # Placeholder for export functionality
    pass

from django.shortcuts import render
from firebase_admin import firestore, auth as firebase_auth
import datetime

db = firestore.client()

@firebase_login_required
def history_page(request):
    # Get date filters from request
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    # Initialize empty lists
    translation_records = []
    search_records = []
    
    # Build query for translations
    translations_query = db.collection("translations").order_by("timestamp", direction=firestore.Query.DESCENDING)
    
    # Apply date filters if provided
    if start_date:
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        translations_query = translations_query.where("timestamp", ">=", start_datetime)
    
    if end_date:
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1)
        translations_query = translations_query.where("timestamp", "<=", end_datetime)
    
    # Get translation documents
    translations_docs = translations_query.stream()
    
    # Process translation records
    for doc in translations_docs:
        data = doc.to_dict()
        user_id = data.get("user_id")
        
        # Try to get email from Firebase Auth
        try:
            user_record = firebase_auth.get_user(user_id)
            email = user_record.email
        except Exception as e:
            email = "Unknown"
            
        translation_records.append({
            "type": "Sign Language to Text",
            "email": email,
            "content": data.get("text", ""),
            "timestamp": data.get("timestamp"),
        })
    
    # Build query for search history
    search_query = db.collection("search_history").order_by("timestamp", direction=firestore.Query.DESCENDING)
    
    # Apply date filters if provided
    if start_date:
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        search_query = search_query.where("timestamp", ">=", start_datetime)
    
    if end_date:
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d') + datetime.timedelta(days=1)
        search_query = search_query.where("timestamp", "<=", end_datetime)
    
    # Get search history documents
    search_docs = search_query.stream()
    
    # Process search records
    for doc in search_docs:
        data = doc.to_dict()
        user_id = data.get("user_id")
        
        # Try to get email from Firebase Auth
        try:
            user_record = firebase_auth.get_user(user_id)
            email = user_record.email
        except Exception as e:
            email = "Unknown"
            
        search_records.append({
            "type": "Text to Sign Language",
            "email": email,
            "content": data.get("query", ""),
            "timestamp": data.get("timestamp"),
        })
    
    # Combine both record types
    all_records = translation_records + search_records
    
    # Sort all records by timestamp (newest first)
    all_records.sort(key=lambda x: x['timestamp'], reverse=True)
    
    context = {"history_records": all_records}
    return render(request, "history.html", context)

@firebase_login_required
def export_analytics(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sign_language_analytics.csv"'

    writer = csv.writer(response)
    
    writer.writerow(['Most Translated Words'])
    writer.writerow(['Rank', 'Word/Phrase', 'Translation Count'])
    
    writer.writerow([])
    writer.writerow(['User Signups by Month'])
    writer.writerow(['Month', 'New Users'])
    
    writer.writerow([])
    writer.writerow(['Most Active Users'])
    writer.writerow(['Rank', 'User', 'Email', 'Translation Count'])
    
    return response

@csrf_exempt
def get_sign_video(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            input_text = data.get('text', '').strip().lower()
            
            words = re.findall(r'\w+', input_text)
            videos = []
            
            for word in words:
                video_words = VideoWord.collection.filter('word', '==', word).fetch()
                if video_words:
                    for video_word in video_words:
                        video = video_word.video.get()
                        videos.append({
                            'word': word,
                            'video_url': video.video_file,
                            'video_title': video.title
                        })
                else:
                    videos.append({'word': word, 'error': 'No sign available'})
            
            return JsonResponse({'status': 'success', 'videos': videos})
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@firebase_login_required
def save_sentence(request):
    if request.method == "POST":
        uid = request.session.get("uid")  # ✅ Firebase UID stored in session
        if not uid:
            messages.error(request, "No user session found. Please log in.")
            return redirect("login")

        translated_text = request.POST.get("sentencetext")
        timestamp = timezone.now()

        db.collection("translations").add({
            "text": translated_text,
            "user_id": uid,  # ✅ now storing correct UID
            "timestamp": timestamp,
            "type": "letters"
        })

        return JsonResponse({"status": "success"})

@firebase_login_required
def get_sentencehistory(request):
    uid = request.session.get("uid")
    if not uid:
        return JsonResponse({"error": "Not logged in"}, status=401)

    # Filter by user_id AND type == "letters"
    docs = db.collection("translations") \
             .where("user_id", "==", uid) \
             .where("type", "==", "letters") \
             .get()

    history = []
    for doc in docs:
        data = doc.to_dict()
        history.append({
            "text": data.get("text"),
            "timestamp": str(data.get("timestamp")) if data.get("timestamp") else None
        })

    return JsonResponse(history, safe=False)




predictor = SignPredictor()

def sign(request):
    return render(request, 'fsltotext.html')

def predict(request):
    # 1. Handle Image Upload (POST)
    # The frontend now sends a snapshot of the video frame here
    if request.method == "POST":
        image_data = request.POST.get('image')
        if image_data:
            # Send the base64 image to predictor.py for processing
            data = predictor.process_web_frame(image_data) 
            return JsonResponse(data)

    # 2. Handle Clear Command (GET)
    if request.GET.get("clear") == "true":
        predictor.clear_fun()
        return JsonResponse({
            'character': '',
            'sentence': '',
            'suggestions': ['', '', '', '']
        })

    # 3. Default GET (Status check)
    return JsonResponse(predictor.get_status())

@csrf_exempt
def apply_suggestion(request):
    data = json.loads(request.body)
    new_sentence = data.get("sentence", "")
    predictor.apply_suggestion(new_sentence)
    return JsonResponse({"success": True})

@firebase_login_required
def detection_page(request):
    return render(request, 'predict.html')

@csrf_exempt
def translate_phrase(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            input_text = data.get('text', '').strip().lower()
            
            video_words = VideoWord.collection.filter('word', '==', input_text).fetch()
            if video_words:
                video = video_words[0].video.get()
                return JsonResponse({'status': 'success', 'video_path': video.video_file})
            else:
                return JsonResponse({'status': 'error', 'message': 'No sign available'}, status=404)
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

# Initialize the predictor
word_predictor = WordPredictor()

# Unused helper (Video feed is now client-side)
active_video_feeds = {}

@csrf_exempt
@require_http_methods(["POST"])
def release_camera(request):
    # Release resources (reset state)
    # predictor.release() # Uncomment if you have the other predictor here
    word_predictor.release()
    return JsonResponse({'status': 'camera_released'})

# --- DELETED: v_feed ---
# We no longer use StreamingHttpResponse because the browser 
# handles the video feed directly via the <video> tag.

@csrf_exempt
def get_prediction(request):
    """
    Handles both receiving frames (POST) and checking status (GET).
    """
    # 1. Handle Image Upload (POST)
    if request.method == "POST":
        image_data = request.POST.get('image')
        if image_data:
            # Send the base64 image to utils.py for processing
            data = word_predictor.process_web_frame(image_data)
            return JsonResponse(data)

    # 2. Handle Status Check (GET)
    # Returns the latest prediction state
    return JsonResponse(word_predictor.get_status())

@csrf_exempt
@require_http_methods(["POST"])
def reset_prediction(request):
    word_predictor.reset()
    return JsonResponse({'status': 'reset'})

# views.py
from django.contrib import messages
from django.shortcuts import redirect
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm

from django.contrib import messages, auth
from django.shortcuts import redirect

from django.contrib.auth.decorators import login_required

@login_required
def change_password(request):
    if request.method == 'POST':
        current = request.POST.get('current_password')
        new = request.POST.get('new_password')
        confirm = request.POST.get('confirm_password')

        if new != confirm:
            messages.error(request, "New passwords do not match.")
            return redirect('account')  

        user = request.user
        if not user.check_password(current):
            messages.error(request, "Current password is incorrect.")
            return redirect('account')

        user.set_password(new)
        user.save()
        auth.update_session_auth_hash(request, user)  # keep logged in
        messages.success(request, "Password changed successfully!")
        return redirect('account')
    
from firebase_admin import auth as firebase_auth
from django.shortcuts import render, redirect
from django.contrib import messages
from firebase_admin import auth as firebase_auth
from django.contrib import messages
from django.shortcuts import redirect

def forgot_password_view(request):
    if request.method == "GET":
        email = request.GET.get("email")  # or from a form input
        if not email:
            messages.error(request, "Please provide an email address.")
            return redirect("login")

        try:
            # ✅ Check if user exists in Firebase Authentication
            firebase_auth.get_user_by_email(email)

            # ✅ Generate reset link
            link = firebase_auth.generate_password_reset_link(email)

            # TODO: send link via email (Django send_mail or similar)
            # send_mail("Reset your password", f"Click here: {link}", "noreply@yourapp.com", [email])

            messages.success(request, "Password reset link sent to your email.")
        except firebase_auth.UserNotFoundError:
            messages.error(request, "No account found with this email.")
        except Exception as e:
            messages.error(request, f"Error sending reset link: {str(e)}")

    return redirect("login")



# views.py
from django.shortcuts import render
from .context_processors import get_all_users


def user_list(request):
    users = get_all_users()
    return render(request, "user_management.html", {"users": users})

from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse

# ✅ backblaze_client is no longer inside utils
from .blackblaze_client import upload_video, generate_temp_url, b2_api

# ✅ firebase_client renamed to firebase_Auth
from .firebase_auth import db

import os, tempfile, requests
from django.conf import settings


from django.contrib import messages
from django.shortcuts import redirect
import tempfile
import os

def upload_video_view(request):
    """Upload video to Backblaze + save metadata to Firestore"""
    context = {}

    if request.method == "POST" and request.FILES.get("video"):
        video = request.FILES["video"]
        title = request.POST.get("title")

        # Save temp file locally
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, video.name)

        with open(temp_path, "wb+") as f:
            for chunk in video.chunks():
                f.write(chunk)

        try:
            # Upload to Backblaze
            upload_info = upload_video(temp_path, video.name)

            # Metadata for Firestore
            metadata = {
                "title": title,
                "file_name": upload_info["file_name"],
                "file_id": upload_info["file_id"],
                "url": upload_info["url"],  # signed temp link
            }

            # Save to Firestore
            db.collection("videos").add(metadata)

            # Success message
            messages.success(request, f"Video '{title}' uploaded successfully!")
            context["uploaded"] = metadata  # pass metadata if you want preview

        except Exception as e:
            messages.error(request, f"Error uploading video: {str(e)}")

        finally:
            # ✅ Always cleanup the temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Render the page again (GET or POST)
    return render(request, "add_video.html", context)


from google.cloud import firestore
from django.shortcuts import render, redirect
from firebase_admin import firestore, auth as firebase_auth
from django.contrib import messages

db = firestore.client()
@firebase_login_required
def search_video_view(request):
    """Search video by title and return signed URL + save query to Firestore only if video exists"""

    user_id = request.session.get("uid")  # 🔹 get current logged-in UID
    query = request.GET.get("q")
    video_url, searched = None, False

    if query:
        searched = True
        results = db.collection("videos").where("title", "==", query).get()

        if results:
            video_doc = results[0].to_dict()
            file_name = video_doc.get("file_name")
            video_url = generate_temp_url(file_name)  # signed URL

            # ✅ Save search only if video found
            search_metadata = {
                "query": query,
                "user_id": user_id,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
            db.collection("search_history").add(search_metadata)
        else:
            messages.warning(request, "No video found for your query.")

    # 🔹 Fetch ALL search history for THIS USER (no order_by → no index needed)
    history_docs = db.collection("search_history").where("user_id", "==", user_id).get()

    # 🔹 Sort manually in Python
    history_list = []
    for doc in history_docs:
        item = doc.to_dict()
        email = "Unknown"

        if item.get("user_id"):
            try:
                user_record = firebase_auth.get_user(item.get("user_id"))
                email = user_record.email
            except Exception:
                pass

        history_list.append({
            "query": item.get("query"),
            "email": email,
            "timestamp": item.get("timestamp")
        })

    # 🔹 Sort by timestamp DESC and keep only 5
    history_list.sort(key=lambda x: x["timestamp"] or 0, reverse=True)
    history_list = history_list[:5]

    # 🔹 Format timestamp for display
    for h in history_list:
        if h["timestamp"]:
            h["timestamp"] = h["timestamp"].strftime("%Y-%m-%d %H:%M")

    return render(request, "texttofsl.html", {
        "video_url": video_url,
        "query": query,
        "searched": searched,
        "history_list": history_list
    })


def stream_video_view(request, file_id):
    """(Optional) Proxy streaming via Django with Authorization header"""
    b2_api.authorize_account(
        "production",
        settings.B2_APPLICATION_KEY_ID,
        settings.B2_APPLICATION_KEY
    )

    # Build API download URL
    download_url = f"https://f000.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId={file_id}"
    headers = {"Authorization": b2_api.account_info.get_account_auth_token()}

    # Handle Range headers for seeking
    range_header = request.headers.get("Range")
    if range_header:
        headers["Range"] = range_header

    resp = requests.get(download_url, headers=headers, stream=True)

    status_code = 206 if range_header else 200
    response = StreamingHttpResponse(
        resp.iter_content(chunk_size=8192),
        status=status_code,
        content_type=resp.headers.get("Content-Type", "video/mp4")
    )

    if "Content-Range" in resp.headers:
        response["Content-Range"] = resp.headers["Content-Range"]
    if "Content-Length" in resp.headers:
        response["Content-Length"] = resp.headers["Content-Length"]

    response["Accept-Ranges"] = "bytes"
    return response