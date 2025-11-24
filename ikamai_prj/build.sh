#!/usr/bin/env bash
# Exit immediately if a command fails
set -o errexit

echo "--- 🛠️  Starting Custom Build Script ---"

# --- PART 1: Install the missing C Library (libenchant) locally ---

# Define a local installation directory
LOCAL_DIR="$RENDER_PROJECT_DIR/.local"
mkdir -p "$LOCAL_DIR"

# Only download and compile if not already present (saves time on future deploys)
if [ ! -f "$LOCAL_DIR/lib/libenchant-2.so" ]; then
    echo "--- 📥 Downloading libenchant source ---"
    # Download version 2.2.15
    curl -L https://github.com/AbiWord/enchant/releases/download/v2.2.15/enchant-2.2.15.tar.gz | tar xz
    
    cd enchant-2.2.15
    
    echo "--- ⚙️  Configuring libenchant ---"
    # Configure to install in our local folder, not system folders
    ./configure --prefix="$LOCAL_DIR" --disable-static
    
    echo "--- 🔨 Compiling libenchant ---"
    make
    make install
    
    cd ..
    rm -rf enchant-2.2.15
else
    echo "--- ✅ libenchant already compiled ---"
fi

# --- PART 2: Setup Environment for the Build ---

# Tell Linux where to find the library we just compiled
export LD_LIBRARY_PATH="$LOCAL_DIR/lib:$LD_LIBRARY_PATH"
# Tell pyenchant specifically where the library is
export PYENCHANT_LIBRARY_PATH="$LOCAL_DIR/lib/libenchant-2.so"

# --- PART 3: Run your original commands ---

echo "--- 🐍 Installing Python requirements ---"
pip install -r requirements.txt

echo "--- 📦 Collecting Static Files ---"
# Note: Based on your logs, manage.py is inside 'ikamai_prj'
python ikamai_prj/manage.py collectstatic --noinput