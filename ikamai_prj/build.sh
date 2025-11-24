#!/usr/bin/env bash
# build.sh

# Install system dependencies for enchant
apt-get update
apt-get install -y enchant-2 libenchant-2-2

pip install -r requirements.txt && python manage.py collectstatic --noinput
