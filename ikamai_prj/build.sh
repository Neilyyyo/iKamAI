#!/usr/bin/env bash
# build.sh

# Install system dependencies for enchant
apt-get update
apt-get install -y enchant-2 libenchant-2-2 libenchant-2-dev

# Install Python dependencies
pip install -r requirements.txt

# Run Django migrations and collect static files
python manage.py migrate
python manage.py collectstatic --noinput