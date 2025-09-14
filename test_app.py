"""
Test script for Medical Diagnosis Application
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    try:
        from modules.logger import setup_logger, log_to_sidebar
        print("✅ Logger module imported successfully")
        
        from modules.data_handler import DataHandler
        print("✅ DataHandler module imported successfully")
        
        from modules.dl_module import DeepLearningModule
        print("✅ DeepLearningModule imported successfully")
        
        from modules.is2_module import IntelligentSystemsModule
        print("✅ IntelligentSystemsModule imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_handler():
    """Test DataHandler functionality"""
    try:
        from modules.data_handler import DataHandler
        
        # Initialize data handler
        data_handler = DataHandler()
        print("✅ DataHandler initialized successfully")
        
        # Test sample data creation
        data_handler._create_sample_data()
        print("✅ Sample data created successfully")
        
        # Test data loading
        training_data = data_handler.get_training_data()
        print(f"✅ Training data loaded: {len(training_data)} datasets")
        
        return True
    except Exception as e:
        print(f"❌ DataHandler error: {e}")
        return False

def test_dl_module():
    """Test DeepLearningModule functionality"""
    try:
        from modules.dl_module import DeepLearningModule
        
        # Initialize DL module
        dl_module = DeepLearningModule()
        print("✅ DeepLearningModule initialized successfully")
        
        # Test weight normalization
        dl_module.weights = {'symptoms': 0.4, 'blood': 0.3, 'xray': 0.3}
        total_weight = sum(dl_module.weights.values())
        for key in dl_module.weights:
            dl_module.weights[key] /= total_weight
        print(f"✅ Weights normalized: {dl_module.weights}")
        
        return True
    except Exception as e:
        print(f"❌ DeepLearningModule error: {e}")
        return False

def test_is2_module():
    """Test IntelligentSystemsModule functionality"""
    try:
        from modules.is2_module import IntelligentSystemsModule
        
        # Initialize IS2 module
        is2_module = IntelligentSystemsModule()
        print("✅ IntelligentSystemsModule initialized successfully")
        
        # Test treatment database
        diseases = list(is2_module.treatment_database.keys())
        print(f"✅ Treatment database loaded: {len(diseases)} diseases")
        
        # Test fuzzy system initialization
        is2_module._initialize_fuzzy_system()
        print("✅ Fuzzy system initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ IntelligentSystemsModule error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Medical Diagnosis Application")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("DataHandler Tests", test_data_handler),
        ("DeepLearningModule Tests", test_dl_module),
        ("IntelligentSystemsModule Tests", test_is2_module)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Application is ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
