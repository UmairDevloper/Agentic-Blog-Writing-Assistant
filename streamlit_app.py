import hmac
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass  # no secrets file when running locally with a .env

password = os.environ.get("APP_PASSWORD")
if password:
    entered = st.text_input("Password", type="password")
    if not hmac.compare_digest(entered, password):
        st.stop()

from seo_blog_assistant.ui import run_ui

run_ui()
