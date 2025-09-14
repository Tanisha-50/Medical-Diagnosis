"""
Logging utility for the Medical Diagnosis Application
"""

import logging
import streamlit as st
from datetime import datetime

def setup_logger():
    """Setup and configure logger for the application"""
    
    # Create logger
    logger = logging.getLogger('medical_diagnosis')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def log_to_sidebar(message, level="INFO"):
    """Log message to Streamlit sidebar"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if level == "ERROR":
        st.sidebar.error(f"❌ {timestamp}: {message}")
    elif level == "WARNING":
        st.sidebar.warning(f"⚠️ {timestamp}: {message}")
    elif level == "SUCCESS":
        st.sidebar.success(f"✅ {timestamp}: {message}")
    else:
        st.sidebar.info(f"ℹ️ {timestamp}: {message}")

def log_to_main(message, level="INFO"):
    """Log message to main Streamlit area"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if level == "ERROR":
        st.error(f"❌ {timestamp}: {message}")
    elif level == "WARNING":
        st.warning(f"⚠️ {timestamp}: {message}")
    elif level == "SUCCESS":
        st.success(f"✅ {timestamp}: {message}")
    else:
        st.info(f"ℹ️ {timestamp}: {message}")
