from fireo.models import Model
from fireo.fields import TextField, DateTime, ReferenceField, NumberField, IDField

class User(Model):
    id = IDField()
    first_name = TextField()
    last_name = TextField()
    email = TextField(required=True)
    username = TextField(required=True)

    class Meta:
        collection_name = "users"

class History(Model):
    user = ReferenceField(User)
    input_text = TextField()
    video_path = TextField()
    timestamp = DateTime(auto_now_add=True)

    class Meta:
        collection_name = "history"

class SignLanguageVideo(Model):
    id = IDField()
    title = TextField()
    video_file = TextField()
    video_path = TextField()

    class Meta:
        collection_name = "sign_language_videos"

class VideoWord(Model):
    video = ReferenceField(SignLanguageVideo)
    word = TextField()

    class Meta:
        collection_name = "video_words"

class SignLanguageHistory(Model):
    user = ReferenceField(User, null=True)
    translation = TextField()
    prediction_confidence = NumberField()
    timestamp = DateTime(auto_now_add=True)

    class Meta:
        collection_name = "sign_language_history"

class Sentence(Model):
    user = ReferenceField(User)
    text = TextField()
    timestamp = DateTime(auto_now_add=True)

    class Meta:
        collection_name = "sentences"