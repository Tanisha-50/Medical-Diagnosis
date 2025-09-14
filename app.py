"""
Medical Diagnosis Application - Main Streamlit App
Doctor's Screen Only - Deep Learning & Intelligent Systems II Modules
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from modules.dl_module import DeepLearningModule
from modules.is2_module import IntelligentSystemsModule
from modules.data_handler import DataHandler
from modules.logger import setup_logger

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Medical Diagnosis System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize logger
    logger = setup_logger()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-header {
        font-size: 1.8rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🏥 Medical Diagnosis System</h1>', unsafe_allow_html=True)
    st.markdown("### Doctor's Interface - Deep Learning & Intelligent Systems II")
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    module = st.sidebar.selectbox(
        "Select Module",
        ["Deep Learning (Disease Diagnosis)", "Intelligent Systems II (Treatment Recommendation)"]
    )
    
    # Initialize data handler
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    
    # Initialize modules
    if 'dl_module' not in st.session_state:
        st.session_state.dl_module = DeepLearningModule()
    
    if 'is2_module' not in st.session_state:
        st.session_state.is2_module = IntelligentSystemsModule()
    
    # Module selection and execution
    if module == "Deep Learning (Disease Diagnosis)":
        st.markdown('<h2 class="module-header">🧠 Deep Learning Module</h2>', unsafe_allow_html=True)
        st.session_state.dl_module.run_module()
        
    elif module == "Intelligent Systems II (Treatment Recommendation)":
        st.markdown('<h2 class="module-header">🎯 Intelligent Systems II Module</h2>', unsafe_allow_html=True)
        st.session_state.is2_module.run_module()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Medical Diagnosis System**")
    st.sidebar.markdown("Version 1.0")
    st.sidebar.markdown("Doctor's Interface Only")

if __name__ == "__main__":
    main()
