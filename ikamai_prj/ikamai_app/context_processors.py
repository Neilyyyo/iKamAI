from firebase_admin import firestore

db = firestore.client()

def firebase_user(request):
    if 'uid' in request.session:
        doc = db.collection('users').document(request.session['uid']).get()
        if doc.exists:
            return {'firebase_user': doc.to_dict()}
    return {'firebase_user': None}


# utils/firebase_utils.py
from firebase_admin import firestore

db = firestore.client()

def get_all_users():
    users_ref = db.collection("users")
    docs = users_ref.stream()

    users = []
    for doc in docs:
        data = doc.to_dict()
        created_at = data.get("created_at")

        # Firestore already gives you a datetime-like object
        if created_at:
            created_at = created_at.replace(tzinfo=None)  # optional, to strip timezone

        users.append({
            # "id": doc.id,
            "first_name": data.get("first_name", ""),
            "last_name": data.get("last_name", ""),
            "email": data.get("email", ""),
            "username": data.get("username", ""),
            "date_created": created_at,
        })

    return users
# utils/firebase_utils.py
from firebase_admin import firestore

db = firestore.client()

def get_translation_history(start_date=None, end_date=None):
    history_ref = db.collection("translations")
    
    # Apply date filter if provided
    if start_date and end_date:
        history_ref = history_ref.where("created_at", ">=", start_date).where("created_at", "<=", end_date)

    docs = history_ref.stream()

    # Pre-fetch all users into a dictionary {user_id: user_data}
    users_ref = db.collection("users").stream()
    users = {user_doc.id: user_doc.to_dict() for user_doc in users_ref}

    history = []
    for doc in docs:
        data = doc.to_dict() or {}

        created_at = data.get("created_at")
        if created_at:
            # Firestore Timestamp acts like datetime
            created_at = created_at.replace(tzinfo=None)

        # Find user by user_id
        users= users.get(data.get("user_id"), {})
        full_name = f"{users.get('first_name', '')} {users.get('last_name', '')}".strip()

        history.append({
            "id": doc.id,
            "user_name": full_name or "Unknown",
            "text": data.get("text", ""),
            "date_created": created_at,
        })

    return history

