# Step 1: Create a virtual environment named `.venv`
python3 -m venv .venv

# Step 2: Activate the virtual environment
source .venv/bin/activate

# Step 3: Install required Python packages
pip install fastapi uvicorn jinja2 sqlalchemy aiosqlite

# Step 4: Save installed packages to requirements.txt
pip freeze > requirements.txt

# Step 5: (Optional) Upgrade pip to the latest version
pip install --upgrade pip

# Step 6: Install all dependencies from requirements.txt (useful if cloning fresh)
pip install -r requirements.txt

# Step 7: Start the FastAPI server with hot-reloading
uvicorn backend.app.main:app --reload

# Youtube demo : https://youtu.be/B2eHtU8NMvk?si=RalgJVCCYc4Vhj9J
