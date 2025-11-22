from django import forms

class VideoForm(forms.Form):
    title = forms.CharField(max_length=100, widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Enter video title'
    }))
    video_file = forms.FileField(widget=forms.FileInput(attrs={
        'class': 'form-control',
        'accept': 'video/mp4,video/webm,video/ogg'
    }))

class WordForm(forms.Form):
    word = forms.CharField(max_length=50, widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Enter the word/phrase this video represents'
    }))
