#!/usr/bin/env bash
# Exit immediately if a command fails
set -o errexit

echo "--- 🛠️  Starting Custom Build Script ---"

# --- PART 1: Install the missing C Library (libenchant) locally ---

# Get the absolute path of the current directory
# This fixes the "expected an absolute directory" error
PROJECT_ROOT=$(pwd)
LOCAL_DIR="$PROJECT_ROOT/.local"

echo "Installing to: $LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

# Only download and compile if not already present
if [ ! -f "$LOCAL_DIR/lib/libenchant-2.so" ]; then
    echo "--- 📥 Downloading libenchant source ---"
    curl -L https://github.com/AbiWord/enchant/releases/download/v2.2.15/enchant-2.2.15.tar.gz | tar xz
    
    cd enchant-2.2.15
    
    echo "--- ⚙️  Configuring libenchant ---"
    # Use the absolute path we calculated above
    ./configure --prefix="$LOCAL_DIR" --disable-static
    
    echo "--- 🔨 Compiling libenchant (This may take 1-2 mins) ---"
    make
    make install
    
    cd ..
    rm -rf enchant-2.2.15
else
    echo "--- ✅ libenchant already compiled ---"
fi

# --- PART 2: Setup Environment for the Build ---

export LD_LIBRARY_PATH="$LOCAL_DIR/lib:$LD_LIBRARY_PATH"
export PYENCHANT_LIBRARY_PATH="$LOCAL_DIR/lib/libenchant-2.so"

# --- PART 3: Run your original commands ---

echo "--- 🐍 Installing Python requirements ---"
pip install -r requirements.txt

echo "--- 📦 Collecting Static Files ---"
python manage.py collectstatic --noinput