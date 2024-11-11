# Server Setup

Follow these steps to set up the server:

```bash
# Update pip to the latest version
python.exe -m pip install --upgrade pip

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Install dependencies from the requirements file
pip install -r requirements.txt

# Run the App
streamlit run main.py
```

Make sure to activate the virtual environment every time you work on this project to ensure all dependencies are correctly managed.

# Future Aspects
- Live graphs
- Pipelines
- Notification system
- Realtime analysis simulation