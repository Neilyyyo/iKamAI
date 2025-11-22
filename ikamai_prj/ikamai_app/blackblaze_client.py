from b2sdk.v2 import InMemoryAccountInfo, B2Api
from django.conf import settings

info = InMemoryAccountInfo()
b2_api = B2Api(info)

# Authorize with credentials
b2_api.authorize_account(
    "production",
    settings.B2_APPLICATION_KEY_ID,
    settings.B2_APPLICATION_KEY
)

# Get bucket
bucket = b2_api.get_bucket_by_name(settings.B2_BUCKET_NAME)

# Base download URL
DOWNLOAD_URL = f"{b2_api.account_info.get_download_url()}/file/{settings.B2_BUCKET_NAME}"


def upload_video(file_path, file_name):
    """Upload video to Backblaze and return file metadata + signed URL."""
    # Upload inside 'videos/' folder
    file_info = bucket.upload_local_file(
        local_file=file_path,
        file_name=f"videos/{file_name}"
    )

    # Generate temporary signed URL (1 hour default)
    url = generate_temp_url(f"videos/{file_name}", duration=3600)

    return {
        "file_name": f"videos/{file_name}",
        "file_id": file_info.id_,
        "url": url
    }


def generate_temp_url(filename, duration=3600):
    """Generate a signed temporary URL for a private file."""
    prefix = filename if "/" in filename else f"videos/{filename}"

    auth_token = bucket.get_download_authorization(
        file_name_prefix=prefix,
        valid_duration_in_seconds=duration
    )

    return f"{DOWNLOAD_URL}/{prefix}?Authorization={auth_token}"
