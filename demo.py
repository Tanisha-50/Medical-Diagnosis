"""
Demonstration script for Medical Diagnosis Application
Shows how to use the application programmatically
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.data_handler import DataHandler
from modules.dl_module import DeepLearningModule
from modules.is2_module import IntelligentSystemsModule

def demo_data_handler():
    """Demonstrate DataHandler functionality"""
    print("🔍 Demonstrating DataHandler...")
    
    # Initialize data handler
    data_handler = DataHandler()
    
    # Load training data
    training_data = data_handler.get_training_data()
    
    print(f"✅ Loaded {len(training_data)} datasets:")
    for dataset_name, dataset in training_data.items():
        if isinstance(dataset, dict):
            print(f"  - {dataset_name}: {len(dataset.get('conditions', []))} conditions")
        elif isinstance(dataset, pd.DataFrame):
            print(f"  - {dataset_name}: {len(dataset)} samples, {len(dataset.columns)} features")
    
    return data_handler

def demo_dl_module():
    """Demonstrate DeepLearningModule functionality"""
    print("\n🧠 Demonstrating DeepLearningModule...")
    
    # Initialize DL module
    dl_module = DeepLearningModule()
    
    # Set weights
    dl_module.weights = {'symptoms': 0.4, 'blood': 0.3, 'xray': 0.3}
    
    # Normalize weights
    total_weight = sum(dl_module.weights.values())
    for key in dl_module.weights:
        dl_module.weights[key] /= total_weight
    
    print(f"✅ Weights set: {dl_module.weights}")
    
    # Simulate training (in real app, this would be done through UI)
    print("✅ DL module initialized and ready for training")
    
    return dl_module

def demo_is2_module():
    """Demonstrate IntelligentSystemsModule functionality"""
    print("\n🎯 Demonstrating IntelligentSystemsModule...")
    
    # Initialize IS2 module
    is2_module = IntelligentSystemsModule()
    
    # Show available diseases
    diseases = list(is2_module.treatment_database.keys())
    print(f"✅ Available diseases: {diseases}")
    
    # Initialize fuzzy system
    is2_module._initialize_fuzzy_system()
    print("✅ Fuzzy system initialized")
    
    # Simulate patient data
    patient_data = {
        'disease': 'Pneumonia',
        'age': 65,
        'weight': 70,
        'blood_pressure': 140,
        'temperature': 38.5,
        'heart_rate': 95,
        'oxygen_saturation': 92
    }
    
    # Perform fuzzy analysis
    severity_score = is2_module._perform_fuzzy_analysis(patient_data)
    print(f"✅ Severity score: {severity_score:.3f}")
    
    # Determine severity level
    if severity_score < 0.3:
        severity_level = "Mild"
    elif severity_score < 0.7:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"
    
    print(f"✅ Severity level: {severity_level}")
    
    return is2_module

def demo_integration():
    """Demonstrate integration between modules"""
    print("\n🔗 Demonstrating Module Integration...")
    
    # Initialize all modules
    data_handler = demo_data_handler()
    dl_module = demo_dl_module()
    is2_module = demo_is2_module()
    
    # Simulate a complete diagnosis workflow
    print("\n📋 Simulating Complete Diagnosis Workflow:")
    
    # 1. Data loading
    print("1. ✅ Data loaded and preprocessed")
    
    # 2. DL module training
    print("2. ✅ Deep Learning models trained")
    
    # 3. Diagnosis simulation
    print("3. 🔍 Simulating diagnosis...")
    
    # Simulate prediction scores
    xray_score = np.random.random(4)
    blood_score = np.random.random(2)
    symptoms_score = np.random.random(10)
    
    # Calculate final scores using weighted fusion
    max_len = max(len(xray_score), len(blood_score), len(symptoms_score))
    xray_padded = np.pad(xray_score, (0, max_len - len(xray_score)))
    blood_padded = np.pad(blood_score, (0, max_len - len(blood_score)))
    symptoms_padded = np.pad(symptoms_score, (0, max_len - len(symptoms_score)))
    
    final_scores = (
        dl_module.weights['xray'] * xray_padded +
        dl_module.weights['blood'] * blood_padded +
        dl_module.weights['symptoms'] * symptoms_padded
    )
    
    # Find predicted disease
    diseases = ['Pneumonia', 'COVID-19', 'Tuberculosis', 'Normal', 'Diabetes', 
                'Hypertension', 'Heart Disease', 'Flu', 'Common Cold', 'Bronchitis']
    predicted_idx = np.argmax(final_scores)
    predicted_disease = diseases[predicted_idx]
    
    print(f"   Predicted Disease: {predicted_disease}")
    print(f"   Confidence Score: {final_scores[predicted_idx]:.3f}")
    
    # 4. IS2 module treatment recommendation
    print("4. 🎯 Treatment recommendation...")
    
    # Set patient data for IS2 module
    patient_data = {
        'disease': predicted_disease,
        'age': 65,
        'weight': 70,
        'blood_pressure': 140,
        'temperature': 38.5,
        'heart_rate': 95,
        'oxygen_saturation': 92
    }
    
    # Perform fuzzy analysis
    severity_score = is2_module._perform_fuzzy_analysis(patient_data)
    print(f"   Severity Score: {severity_score:.3f}")
    
    # Show treatment options
    if predicted_disease in is2_module.treatment_database:
        treatments = is2_module.treatment_database[predicted_disease]
        print(f"   Available Treatments: {len(treatments)}")
        for i, treatment in enumerate(treatments, 1):
            print(f"     {i}. {treatment['name']}")
    
    print("✅ Complete workflow demonstrated successfully!")

def main():
    """Main demonstration function"""
    print("🏥 Medical Diagnosis Application - Demonstration")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demo_integration()
        
        print("\n" + "=" * 60)
        print("🎉 Demonstration completed successfully!")
        print("\nTo run the full Streamlit application:")
        print("streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
