"""
Data Handler for Medical Diagnosis Application
Handles downloading, loading, and preprocessing of medical datasets
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import requests
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
from modules.logger import log_to_sidebar, log_to_main

class DataHandler:
    """Handles all data operations for the medical diagnosis system"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset paths
        self.datasets = {
            'chest_xray': self.data_dir / "chest_xray",
            'diabetes': self.data_dir / "diabetes.csv",
            'symptoms': self.data_dir / "symptoms.csv"
        }
        
        # Data storage
        self.chest_xray_data = None
        self.diabetes_data = None
        self.symptoms_data = None
        
        # Preprocessing objects
        self.scalers = {}
        self.encoders = {}
    
    def download_datasets(self):
        """Download datasets from Kaggle (placeholder - requires Kaggle API setup)"""
        log_to_sidebar("Starting dataset download process...")
        
        # For now, we'll create sample data structures
        # In production, this would download from Kaggle
        self._create_sample_data()
        
        log_to_sidebar("Dataset preparation completed", "SUCCESS")
    
    def _create_sample_data(self):
        """Create sample data for demonstration purposes"""
        log_to_sidebar("Creating sample datasets for demonstration...")
        
        # Create sample chest X-ray data structure
        self._create_sample_chest_xray()
        
        # Create sample diabetes data
        self._create_sample_diabetes()
        
        # Create sample symptoms data
        self._create_sample_symptoms()
        
        log_to_sidebar("Sample datasets created successfully", "SUCCESS")
    
    def _create_sample_chest_xray(self):
        """Create sample chest X-ray dataset structure"""
        # Create directory structure
        chest_dir = self.datasets['chest_xray']
        chest_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different conditions
        conditions = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        for condition in conditions:
            (chest_dir / condition).mkdir(exist_ok=True)
        
        # Generate sample images (placeholder)
        log_to_sidebar("Created chest X-ray dataset structure")
    
    def _create_sample_diabetes(self):
        """Create sample diabetes dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.randint(20, 80, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'blood_pressure': np.random.normal(120, 20, n_samples),
            'glucose': np.random.normal(100, 30, n_samples),
            'insulin': np.random.normal(50, 20, n_samples),
            'hba1c': np.random.normal(5.5, 1.5, n_samples),
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        df = pd.DataFrame(data)
        df.to_csv(self.datasets['diabetes'], index=False)
        log_to_sidebar("Created sample diabetes dataset")
    
    def _create_sample_symptoms(self):
        """Create sample symptoms dataset"""
        np.random.seed(42)
        n_samples = 1500
        
        symptoms = [
            'fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue',
            'headache', 'nausea', 'vomiting', 'diarrhea', 'abdominal_pain',
            'joint_pain', 'muscle_weakness', 'dizziness', 'confusion', 'rash'
        ]
        
        diseases = [
            'Pneumonia', 'COVID-19', 'Tuberculosis', 'Bronchitis', 'Asthma',
            'Diabetes', 'Hypertension', 'Heart Disease', 'Flu', 'Common Cold'
        ]
        
        data = {}
        for symptom in symptoms:
            data[symptom] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        
        data['disease'] = np.random.choice(diseases, n_samples)
        data['severity'] = np.random.choice(['Mild', 'Moderate', 'Severe'], n_samples)
        data['age'] = np.random.randint(18, 80, n_samples)
        data['gender'] = np.random.choice(['Male', 'Female'], n_samples)
        
        df = pd.DataFrame(data)
        df.to_csv(self.datasets['symptoms'], index=False)
        log_to_sidebar("Created sample symptoms dataset")
    
    def load_chest_xray_data(self):
        """Load and preprocess chest X-ray data"""
        log_to_sidebar("Loading chest X-ray data...")
        
        # For demonstration, we'll create a sample structure
        # In production, this would load actual images
        self.chest_xray_data = {
            'images': [],
            'labels': [],
            'conditions': ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
        }
        
        log_to_sidebar("Chest X-ray data loaded", "SUCCESS")
        return self.chest_xray_data
    
    def load_diabetes_data(self):
        """Load and preprocess diabetes data"""
        log_to_sidebar("Loading diabetes data...")
        
        if not self.datasets['diabetes'].exists():
            self._create_sample_diabetes()
        
        self.diabetes_data = pd.read_csv(self.datasets['diabetes'])
        
        # Preprocess
        self.diabetes_data = self._preprocess_diabetes_data(self.diabetes_data)
        
        log_to_sidebar("Diabetes data loaded", "SUCCESS")
        return self.diabetes_data
    
    def load_symptoms_data(self):
        """Load and preprocess symptoms data"""
        log_to_sidebar("Loading symptoms data...")
        
        if not self.datasets['symptoms'].exists():
            self._create_sample_symptoms()
        
        self.symptoms_data = pd.read_csv(self.datasets['symptoms'])
        
        # Preprocess
        self.symptoms_data = self._preprocess_symptoms_data(self.symptoms_data)
        
        log_to_sidebar("Symptoms data loaded", "SUCCESS")
        return self.symptoms_data
    
    def _preprocess_diabetes_data(self, df):
        """Preprocess diabetes dataset"""
        # Handle categorical variables
        le_gender = LabelEncoder()
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        self.encoders['diabetes_gender'] = le_gender
        
        # Scale numerical features
        numerical_features = ['age', 'bmi', 'blood_pressure', 'glucose', 'insulin', 'hba1c']
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        self.scalers['diabetes'] = scaler
        
        return df
    
    def _preprocess_symptoms_data(self, df):
        """Preprocess symptoms dataset"""
        # Handle categorical variables
        le_disease = LabelEncoder()
        le_severity = LabelEncoder()
        le_gender = LabelEncoder()
        
        df['disease_encoded'] = le_disease.fit_transform(df['disease'])
        df['severity_encoded'] = le_severity.fit_transform(df['severity'])
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        
        self.encoders['symptoms_disease'] = le_disease
        self.encoders['symptoms_severity'] = le_severity
        self.encoders['symptoms_gender'] = le_gender
        
        # Scale age
        scaler = StandardScaler()
        df['age_scaled'] = scaler.fit_transform(df[['age']])
        self.scalers['symptoms'] = scaler
        
        return df
    
    def get_training_data(self):
        """Get all training data for model training"""
        log_to_sidebar("Preparing training data...")
        
        # Load all datasets
        chest_data = self.load_chest_xray_data()
        diabetes_data = self.load_diabetes_data()
        symptoms_data = self.load_symptoms_data()
        
        return {
            'chest_xray': chest_data,
            'diabetes': diabetes_data,
            'symptoms': symptoms_data
        }
    
    def preprocess_doctor_input(self, input_type, data):
        """Preprocess doctor's input data for prediction"""
        log_to_sidebar(f"Preprocessing {input_type} input...")
        
        if input_type == 'diabetes':
            # Preprocess diabetes input
            processed_data = self._preprocess_diabetes_input(data)
        elif input_type == 'symptoms':
            # Preprocess symptoms input
            processed_data = self._preprocess_symptoms_input(data)
        elif input_type == 'chest_xray':
            # Preprocess chest X-ray input
            processed_data = self._preprocess_chest_xray_input(data)
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        log_to_sidebar(f"{input_type} input preprocessed", "SUCCESS")
        return processed_data
    
    def _preprocess_diabetes_input(self, data):
        """Preprocess diabetes input data"""
        # Convert to DataFrame if not already
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Apply same preprocessing as training data
        if 'gender' in df.columns:
            df['gender_encoded'] = self.encoders['diabetes_gender'].transform(df['gender'])
        
        numerical_features = ['age', 'bmi', 'blood_pressure', 'glucose', 'insulin', 'hba1c']
        df[numerical_features] = self.scalers['diabetes'].transform(df[numerical_features])
        
        return df
    
    def _preprocess_symptoms_input(self, data):
        """Preprocess symptoms input data"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Apply same preprocessing as training data
        if 'disease' in df.columns:
            df['disease_encoded'] = self.encoders['symptoms_disease'].transform(df['disease'])
        if 'severity' in df.columns:
            df['severity_encoded'] = self.encoders['symptoms_severity'].transform(df['severity'])
        if 'gender' in df.columns:
            df['gender_encoded'] = self.encoders['symptoms_gender'].transform(df['gender'])
        
        if 'age' in df.columns:
            df['age_scaled'] = self.scalers['symptoms'].transform(df[['age']])
        
        return df
    
    def _preprocess_chest_xray_input(self, image):
        """Preprocess chest X-ray image input"""
        # Resize image to standard size
        target_size = (224, 224)
        
        if isinstance(image, str):
            # Load image from file path
            img = cv2.imread(image)
        else:
            # Image is already loaded
            img = image
        
        # Convert to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
